"""Check kuma against pyarrow across the Arrow IPC stream format.

This is driven by TestPyarrowStream and takes a directory as its only argument.

The schema check and the batch check next door pass one message at a time, which
leaves the framing between the messages untested: a stream is a schema, a run of
batches and a marker saying there are no more, and the only thing that can say
kuma puts those together the way everybody else does is somebody else's reader.

kuma has already written go-NAME.arrows and go.txt, which is one line per case
holding the name and the rendering pyarrow should produce for the whole stream.
This opens each of them, checks the rendering, and writes the batches back out
as back-NAME.arrows with pyarrow's own writer. Then it writes its own cases the
same way, along with py.txt saying what kuma should read.

A rendering is the same one the batch check uses, with the batches of a stream
joined by |.
"""

import os
import sys

import pyarrow as pa

from batch import render


def read_go(directory):
    """Read the streams kuma wrote and write them back out."""
    problems = []
    count = 0
    manifest = open(os.path.join(directory, "go.txt"), encoding="utf-8")
    with manifest:
        for line in manifest:
            name, want = line.rstrip("\n").split("\t")

            path = os.path.join(directory, "go-%s.arrows" % name)
            try:
                with pa.OSFile(path, "rb") as f:
                    reader = pa.ipc.open_stream(f)
                    schema = reader.schema
                    batches = list(reader)
            except Exception as err:  # noqa: BLE001
                problems.append("%s: %s" % (name, err))
                continue

            count += len(batches)
            got = []
            for batch in batches:
                batch.validate(full=True)
                got.append(render(batch))
            if "|".join(got) != want:
                problems.append("%s: pyarrow reads it as\n  %s\nwant\n  %s"
                                % (name, "|".join(got), want))
                continue

            back = os.path.join(directory, "back-%s.arrows" % name)
            with pa.OSFile(back, "wb") as f:
                writer = pa.ipc.new_stream(f, schema)
                for batch in batches:
                    writer.write_batch(batch)
                writer.close()
    return count, problems


# The streams that start in pyarrow. These are the shapes kuma does not write:
# text as offsets rather than views, a stream with no batches in it at all, and
# a stream whose batches were sliced out of a longer one.
def python_cases():
    long_value = "y" * 40
    text = pa.schema([
        pa.field("string", pa.string()),
        pa.field("binary", pa.binary()),
    ])
    ints = pa.schema([pa.field("id", pa.int64())])

    sliced = pa.record_batch([
        pa.array(list(range(8))),
        pa.array(["a", "bb", None, "dddd", "e", long_value, "g", "hh"]),
    ], names=["id", "text"])

    return [
        ("offsets", text, [
            pa.record_batch([
                pa.array(["a", "bb", None, "dddd"]),
                pa.array([b"\x01", None, b"", b"\x04\x05\x06"],
                         type=pa.binary()),
            ], schema=text),
            pa.record_batch([
                pa.array([long_value, "e"]),
                pa.array([b"", b"\x07"], type=pa.binary()),
            ], schema=text),
        ]),

        ("many", ints, [
            pa.record_batch([pa.array([1, 2, 3], type=pa.int64())],
                            schema=ints),
            pa.record_batch([pa.array([], type=pa.int64())], schema=ints),
            pa.record_batch([pa.array([4], type=pa.int64())], schema=ints),
        ]),

        ("nothing", ints, []),

        ("sliced", sliced.schema, [sliced.slice(3, 4), sliced.slice(1, 1)]),
    ]


def write_python(directory):
    """Write the streams that start in pyarrow, and what kuma should read."""
    lines = []
    written = 0
    for name, schema, batches in python_cases():
        path = os.path.join(directory, "py-%s.arrows" % name)
        with pa.OSFile(path, "wb") as f:
            writer = pa.ipc.new_stream(f, schema)
            for batch in batches:
                writer.write_batch(batch)
            writer.close()
        rendered = [render(batch) for batch in batches]
        lines.append("%s\t%s\n" % (name, "|".join(rendered)))
        written += len(batches)

    path = os.path.join(directory, "py.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return written


def main():
    if len(sys.argv) != 2:
        print("usage: stream.py DIRECTORY", file=sys.stderr)
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
