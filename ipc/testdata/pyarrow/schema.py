"""Check kuma against pyarrow across the Arrow IPC schema message.

This is driven by TestPyarrowSchema and takes a directory as its only argument.
Both sides write schemas into it, and both sides write down what they expect the
other to make of them.

kuma has already written go-NAME.arrow and go.txt, which is one line per case
holding the name and the rendering pyarrow should produce. This reads each of
them, checks the rendering, and writes the schema straight back out as
back-NAME.arrow for kuma to read again. Then it writes its own cases as
py-NAME.arrow and py.txt, which is the same thing the other way around: the
rendering kuma should produce, in kuma's vocabulary.

A rendering is fields separated by |, each of them a name, a type and either
null or notnull separated by :, with metadata following as ;key=value. The
metadata of the schema itself is a last field called @.
"""

import os
import sys

import pyarrow as pa


def render(schema):
    """The schema as the cross check writes one, in pyarrow's vocabulary."""
    parts = []
    for field in schema:
        null = "null" if field.nullable else "notnull"
        parts.append("%s:%s:%s%s" % (field.name, field.type, null,
                                     render_metadata(field.metadata)))
    if schema.metadata:
        parts.append("@" + render_metadata(schema.metadata))
    return "|".join(parts)


def render_metadata(metadata):
    if not metadata:
        return ""
    return "".join(";%s=%s" % (k.decode(), v.decode())
                   for k, v in metadata.items())


def read_go(directory):
    """Read what kuma wrote and write each schema back out."""
    problems = []
    manifest = open(os.path.join(directory, "go.txt"), encoding="utf-8")
    count = 0
    with manifest:
        for line in manifest:
            name, want = line.rstrip("\n").split("\t")
            count += 1

            path = os.path.join(directory, "go-%s.arrow" % name)
            with open(path, "rb") as f:
                buf = pa.py_buffer(f.read())

            try:
                schema = pa.ipc.read_schema(buf)
            except Exception as err:  # noqa: BLE001
                problems.append("%s: %s" % (name, err))
                continue

            got = render(schema)
            if got != want:
                problems.append("%s: pyarrow reads it as\n  %s\nwant\n  %s"
                                % (name, got, want))
                continue

            back = os.path.join(directory, "back-%s.arrow" % name)
            with open(back, "wb") as f:
                f.write(schema.serialize().to_pybytes())
    return count, problems


# The cases that start in pyarrow, each a name, a schema, and the rendering kuma
# should produce for it in kuma's own vocabulary. A rendering of !error means
# kuma has no type for the column and has to say so rather than guess.
#
# These are the schemas kuma cannot write. Every text layout it collapses into
# one, every type it has no equivalent for, and the fields whose defaults it
# always writes out are here, because a reader that has only read its own
# writing has not been tested against anything.
def python_cases():
    long_string = pa.field("s", pa.string())
    return [
        ("offsets", pa.schema([
            pa.field("string", pa.string()),
            pa.field("binary", pa.binary()),
            pa.field("large string", pa.large_string()),
            pa.field("large binary", pa.large_binary()),
            pa.field("string view", pa.string_view()),
            pa.field("binary view", pa.binary_view()),
        ]), "string:string:null|binary:binary:null|large string:string:null|"
            "large binary:binary:null|string view:string:null|"
            "binary view:binary:null"),

        ("defaults", pa.schema([
            pa.field("date", pa.date64(), nullable=False),
            pa.field("time", pa.time32("ms"), nullable=False),
            pa.field("duration", pa.duration("ms"), nullable=False),
            pa.field("interval", pa.month_day_nano_interval(), nullable=False),
            pa.field("timestamp", pa.timestamp("s"), nullable=False),
            pa.field("decimal", pa.decimal128(10, 3), nullable=False),
        ]), "date:date64:notnull|time:time32[ms]:notnull|"
            "duration:duration[ms]:notnull|interval:interval[month_day_nano]:notnull|"
            "timestamp:timestamp[s]:notnull|decimal:decimal128(10, 3):notnull"),

        ("nested", pa.schema([
            pa.field("list", pa.list_(pa.int32())),
            pa.field("map", pa.map_(pa.string(), pa.int64())),
            pa.field("struct", pa.struct([
                pa.field("a", pa.int64(), nullable=False),
                pa.field("b", pa.list_(long_string)),
            ])),
        ]), "list:list<int32>:null|map:map<string, int64>:null|"
            "struct:struct<a: int64 not null, b: list<string>>:null"),

        ("dictionary", pa.schema([
            pa.field("d", pa.dictionary(pa.int16(), pa.string())),
            pa.field("ordered", pa.dictionary(pa.int8(), pa.string(),
                                              ordered=True)),
        ]), "d:dictionary<int16, string>:null|"
            "ordered:dictionary<int8, string>:null"),

        ("metadata", pa.schema([
            pa.field("one", pa.int64(), metadata={"unit": "meters"}),
        ], metadata={"written by": "pyarrow"}),
            "one:int64:null;unit=meters|@;written by=pyarrow"),

        # The types kuma has no equivalent for. Each of these is a real Arrow
        # type that a real producer writes, and reading one has to fail with the
        # name of the column rather than produce something else.
        ("float16", pa.schema([pa.field("half", pa.float16())]), "!error"),
        ("union", pa.schema([
            pa.field("u", pa.dense_union([pa.field("a", pa.int64())])),
        ]), "!error"),
        ("list view", pa.schema([
            pa.field("lv", pa.list_view(pa.int32())),
        ]), "!error"),
        ("run end encoded", pa.schema([
            pa.field("r", pa.run_end_encoded(pa.int32(), pa.int64())),
        ]), "!error"),
    ]


def write_python(directory):
    """Write the cases that start in pyarrow, along with what kuma should read."""
    lines = []
    for name, schema, want in python_cases():
        path = os.path.join(directory, "py-%s.arrow" % name)
        with open(path, "wb") as f:
            f.write(schema.serialize().to_pybytes())
        lines.append("%s\t%s\n" % (name, want))

    path = os.path.join(directory, "py.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return len(lines)


def main():
    if len(sys.argv) != 2:
        print("usage: schema.py DIRECTORY", file=sys.stderr)
        return 2

    directory = sys.argv[1]
    read, problems = read_go(directory)
    written = write_python(directory)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("pyarrow %s, %d schemas out of kuma, %d schemas into it"
          % (pa.__version__, read, written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
