"""Check kuma against pyarrow across the Arrow IPC file format.

This is driven by TestPyarrowFile and takes a directory as its only argument.

The stream check next door passes whole streams, which leaves out the two things
a file adds: the magic at both ends and the footer holding a block per batch. A
footer whose blocks are eight bytes out reads perfectly in the writer that
produced it and nowhere else, so the check worth running is another reader
opening the file and asking for a batch by number.

kuma has already written go-NAME.arrow and go.txt, which is one line per case
holding the name and the rendering pyarrow should produce for the whole file.
This opens each of them, reads the batches backwards so that the footer is what
finds them, checks the rendering, and writes them back out as back-NAME.arrow
with pyarrow's own writer. Then it writes its own cases the same way, along with
py.txt saying what kuma should read.

A rendering is the same one the batch check uses, with the batches of a file
joined by |.
"""

import os
import sys

import pyarrow as pa

from batch import render
from stream import python_cases


def read_go(directory):
    """Read the files kuma wrote and write them back out."""
    problems = []
    count = 0
    manifest = open(os.path.join(directory, "go.txt"), encoding="utf-8")
    with manifest:
        for line in manifest:
            name, want = line.rstrip("\n").split("\t")

            path = os.path.join(directory, "go-%s.arrow" % name)
            try:
                with pa.OSFile(path, "rb") as f:
                    reader = pa.ipc.open_file(f)
                    schema = reader.schema
                    last = reader.num_record_batches - 1
                    batches = [reader.get_batch(i) for i in range(last, -1, -1)]
                batches.reverse()
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

            back = os.path.join(directory, "back-%s.arrow" % name)
            with pa.OSFile(back, "wb") as f:
                writer = pa.ipc.new_file(f, schema)
                for batch in batches:
                    writer.write_batch(batch)
                writer.close()
    return count, problems


def write_python(directory):
    """Write the files that start in pyarrow, and what kuma should read.

    The cases are the stream check's cases, since the shapes kuma does not write
    are the same shapes whichever format they arrive in.
    """
    lines = []
    written = 0
    for name, schema, batches in python_cases():
        path = os.path.join(directory, "py-%s.arrow" % name)
        with pa.OSFile(path, "wb") as f:
            writer = pa.ipc.new_file(f, schema)
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
        print("usage: file.py DIRECTORY", file=sys.stderr)
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
