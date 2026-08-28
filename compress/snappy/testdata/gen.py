"""Write the snappy blocks the decoder tests read.

A decompressor is only worth anything if it agrees with the compressor
everybody else uses, so the blocks here are written by the snappy that ships
inside Arrow, which is Google's own, and read back by kuma. Writing them with a
compressor of our own would only prove that the two halves of one guess agree.

Each case is two files with the same name: the .raw is what went in and the
.snappy is what came out. The test walks the .snappy files, decompresses each
one and expects the .raw beside it.

Run it from this directory with pyarrow installed:

    python3 gen.py

Keep the files small. Between them they have to reach every shape of tag the
compressor emits, and after that a larger file says nothing a smaller one has
not said already.
"""

import pathlib
import random

import pyarrow as pa

# The cases, each a name and the bytes that go in.
#
# Between them they cover a literal short enough to have its length in the tag,
# one long enough to need a byte behind it and one long enough to need two, a
# copy of every length the short form can hold, a copy far enough back to need
# the two byte offset, and a run written as a copy that overlaps itself.
def cases():
    yield "empty", b""
    yield "byte", b"x"
    yield "hello", b"hello hello hello hello world"

    # A literal and nothing else, since random bytes have nothing in them to
    # point back at. Three sizes: one whose length fits in the tag byte, one
    # that needs a byte behind it and one that needs two.
    rng = random.Random(20260828)
    yield "random-small", bytes(rng.randrange(256) for _ in range(40))
    yield "random-medium", bytes(rng.randrange(256) for _ in range(200))
    yield "random-large", bytes(rng.randrange(256) for _ in range(3000))

    # One byte over and over, which is the overlapping copy, and then a short
    # pattern over and over, which is the same thing at a wider offset.
    yield "run", b"\x00" * 5000
    yield "pattern", b"ab" * 40 + b"abcabcabc" * 500

    # Lines that share most of their text with the lines before them, which is
    # what a real column of keys looks like and what snappy is good at.
    keys = "".join(f"customer/2026/08/{i:06d}\n" for i in range(1000))
    yield "keys", keys.encode()

    # Text with words that come back after a few hundred bytes, which is the
    # copy that needs a two byte offset.
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "and", "then", "walks", "back", "again", "slowly"]
    text = " ".join(rng.choice(words) for _ in range(2000))
    yield "text", text.encode()

    # A column of nulls next to a column of values, roughly the way a page of
    # them sits in a parquet file: long runs of nought with real bytes between.
    mixed = bytearray()
    for i in range(200):
        mixed += b"\x00" * (i % 37)
        mixed += bytes([i % 256]) * 3
        mixed += b"value-%d;" % i
    yield "mixed", bytes(mixed)


if __name__ == "__main__":
    here = pathlib.Path(".")
    for old in sorted(here.glob("*.raw")) + sorted(here.glob("*.snappy")):
        old.unlink()

    for name, raw in cases():
        block = pa.compress(raw, codec="snappy").to_pybytes()
        assert pa.decompress(block, len(raw), codec="snappy").to_pybytes() == raw
        (here / f"{name}.raw").write_bytes(raw)
        (here / f"{name}.snappy").write_bytes(block)
        print(f"{name}: {len(raw)} bytes in, {len(block)} out")
