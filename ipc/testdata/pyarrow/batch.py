"""Check kuma against pyarrow across the Arrow IPC record batch message.

This is driven by TestPyarrowBatch and takes a directory as its only argument.
Both sides write batches into it, and both sides write down the values they say
are in them.

The schema check next door passes types across. This one passes values, and a
value read out of the wrong place in the body still looks like a value, so the
only way to catch it is to have somebody else say what the numbers are.

kuma has already written go-NAME-schema.arrow, go-NAME-INDEX.arrow and go.txt,
which is one line per case holding the name, how many batches there are, and the
rendering pyarrow should produce for them. This reads each of them, checks the
rendering, and writes each batch straight back out as back-NAME-INDEX.arrow for
kuma to read again. Then it writes its own cases the same way, along with py.txt
saying what kuma should read.

Every file here holds one message on its own, which is what serialize and
read_record_batch pass around, so neither side needs the stream framing yet.

A rendering is columns separated by ;, each of them a name, an equals sign and
its values separated by commas. A missing value is null, a bool is true or
false, bytes are hex, and a timestamp or a date is the number it is stored as
rather than the moment it means.
"""

import os
import sys

import pyarrow as pa


def render(batch):
    """The batch as the cross check writes one, values only."""
    parts = []
    for name, column in zip(batch.schema.names, batch.columns):
        values = ",".join(render_value(column, i) for i in range(len(column)))
        parts.append("%s=%s" % (name, values))
    return ";".join(parts)


def render_value(column, i):
    """One value, written the way both sides agree to write it."""
    value = column[i]
    if not value.is_valid:
        return "null"

    kind = column.type
    # The bool check comes first because pyarrow hands back a Python bool, and a
    # bool is an int, so any test for a number would take it as one.
    if pa.types.is_boolean(kind):
        return "true" if value.as_py() else "false"
    if (pa.types.is_string(kind) or pa.types.is_large_string(kind)
            or pa.types.is_string_view(kind)):
        return value.as_py()
    if (pa.types.is_binary(kind) or pa.types.is_large_binary(kind)
            or pa.types.is_binary_view(kind)
            or pa.types.is_fixed_size_binary(kind)):
        return value.as_py().hex()
    if pa.types.is_floating(kind):
        return repr(value.as_py())
    return str(stored(column, i))


def stored(column, i):
    """The number a column holds, rather than the day or the moment it means.

    A timestamp far enough out has no Python datetime to become, and the test
    fills the fixed width columns with counting bytes on purpose, so the two
    temporal types are read as the integers they are stored as.
    """
    kind = column.type
    if pa.types.is_timestamp(kind) or pa.types.is_duration(kind):
        return column.view(pa.int64())[i].as_py()
    if pa.types.is_date32(kind) or pa.types.is_time32(kind):
        return column.view(pa.int32())[i].as_py()
    return column[i].as_py()


def read_go(directory):
    """Read what kuma wrote and write each batch back out."""
    problems = []
    count = 0
    manifest = open(os.path.join(directory, "go.txt"), encoding="utf-8")
    with manifest:
        for line in manifest:
            name, batches, want = line.rstrip("\n").split("\t")
            count += int(batches)

            path = os.path.join(directory, "go-%s-schema.arrow" % name)
            with open(path, "rb") as f:
                schema = pa.ipc.read_schema(pa.py_buffer(f.read()))

            got = []
            failed = False
            for i in range(int(batches)):
                path = os.path.join(directory, "go-%s-%d.arrow" % (name, i))
                with open(path, "rb") as f:
                    buf = pa.py_buffer(f.read())
                try:
                    batch = pa.ipc.read_record_batch(buf, schema)
                    batch.validate(full=True)
                except Exception as err:  # noqa: BLE001
                    problems.append("%s batch %d: %s" % (name, i, err))
                    failed = True
                    break
                got.append(render(batch))

                back = os.path.join(directory, "back-%s-%d.arrow" % (name, i))
                with open(back, "wb") as f:
                    f.write(batch.serialize().to_pybytes())

            if failed:
                continue
            if "|".join(got) != want:
                problems.append("%s: pyarrow reads it as\n  %s\nwant\n  %s"
                                % (name, "|".join(got), want))
    return count, problems


# The cases that start in pyarrow, each a name and the batches to write. The
# rendering kuma should produce is taken from the batches themselves, since both
# sides write a rendering the same way and pyarrow is the one being believed
# here.
#
# These are the batches kuma does not write. Almost nothing else in the world
# stores text as views, so the offset layouts are the ones a real file arrives
# in, and a batch sliced out of a longer one is the one whose offsets do not
# start at zero.
def python_cases():
    text = pa.schema([
        pa.field("string", pa.string()),
        pa.field("binary", pa.binary()),
        pa.field("large string", pa.large_string()),
        pa.field("large binary", pa.large_binary()),
    ])
    ints = pa.schema([pa.field("id", pa.int64())])

    long_value = "y" * 40
    sliced = pa.record_batch([
        pa.array(list(range(8))),
        pa.array(["a", "bb", None, "dddd", "e", long_value, "g", "hh"]),
    ], names=["id", "text"])

    return [
        ("offsets", [pa.record_batch([
            pa.array(["a", "bb", None, "dddd"]),
            pa.array([b"\x01", None, b"", b"\x04\x05\x06"], type=pa.binary()),
            pa.array([None, "ee", long_value, "g"], type=pa.large_string()),
            pa.array([b"\x07\x08", b"", None, b"\x09"],
                     type=pa.large_binary()),
        ], schema=text)]),

        ("views", [pa.record_batch([
            pa.array(["a", None, long_value, "dddd"], type=pa.string_view()),
            pa.array([b"\x01\x02", long_value.encode(), None, b""],
                     type=pa.binary_view()),
        ], names=["string view", "binary view"])]),

        ("flat", [pa.record_batch([
            pa.array([1, None, 3], type=pa.int64()),
            pa.array([1.5, 2.25, None], type=pa.float64()),
            pa.array([True, False, None], type=pa.bool_()),
            pa.array([0, 1000, None], type=pa.timestamp("us", tz="UTC")),
            pa.array([0, 1, 2], type=pa.date32()),
            pa.array([b"abc", b"def", b"ghi"], type=pa.binary(3)),
        ], names=["id", "price", "live", "at", "day", "key"])]),

        ("many", [
            pa.record_batch([pa.array([1, 2, 3], type=pa.int64())],
                            schema=ints),
            pa.record_batch([pa.array([], type=pa.int64())], schema=ints),
            pa.record_batch([pa.array([4], type=pa.int64())], schema=ints),
        ]),

        ("empty", [pa.record_batch([
            pa.array([], type=pa.string()),
            pa.array([], type=pa.int64()),
        ], names=["text", "id"])]),

        # A slice writes the values it covers and nothing in front of them, so
        # its offsets start somewhere other than zero and its validity bitmap
        # starts somewhere other than the first bit of a byte.
        ("sliced", [sliced.slice(3, 4), sliced.slice(1, 1)]),

        ("nothing", [pa.record_batch([
            pa.array([None, None, None], type=pa.null()),
            pa.array([7, 8, 9], type=pa.int64()),
        ], names=["nothing", "id"])]),
    ]


def write_python(directory):
    """Write the cases that start in pyarrow, along with what kuma should read."""
    lines = []
    written = 0
    for name, batches in python_cases():
        path = os.path.join(directory, "py-%s-schema.arrow" % name)
        with open(path, "wb") as f:
            f.write(batches[0].schema.serialize().to_pybytes())

        rendered = []
        for i, batch in enumerate(batches):
            path = os.path.join(directory, "py-%s-%d.arrow" % (name, i))
            with open(path, "wb") as f:
                f.write(batch.serialize().to_pybytes())
            rendered.append(render(batch))
        lines.append("%s\t%d\t%s\n" % (name, len(batches), "|".join(rendered)))
        written += len(batches)

    path = os.path.join(directory, "py.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return written


def main():
    if len(sys.argv) != 2:
        print("usage: batch.py DIRECTORY", file=sys.stderr)
        return 2

    directory = sys.argv[1]
    read, problems = read_go(directory)
    written = write_python(directory)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("pyarrow %s, %d batches out of kuma, %d batches into it"
          % (pa.__version__, read, written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
