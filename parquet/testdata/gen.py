"""Write the parquet files the footer tests read.

The files are checked in rather than written by the tests, because the tests run
on every platform in the matrix and pyarrow is only installed on one of them. A
reader has to be checked against files somebody else wrote, so these are written
by pyarrow and read by kuma, and this script is here so that anybody can see
what produced them and write them again.

Run it from this directory with pyarrow installed:

    python3 gen.py

Keep the files small. They are read for what they say about themselves rather
than for the data in them, and nobody is measuring anything with them.
"""

import datetime
import decimal

import pyarrow as pa
import pyarrow.parquet as pq


def alltypes():
    """One row group holding a column of every type the footer has to describe."""
    schema = pa.schema(
        [
            pa.field("flag", pa.bool_(), nullable=False),
            pa.field("small", pa.int8()),
            pa.field("count", pa.int32()),
            pa.field("total", pa.int64()),
            pa.field("unsigned", pa.uint32()),
            pa.field("ratio", pa.float32()),
            pa.field("weight", pa.float64()),
            pa.field("name", pa.string()),
            pa.field("blob", pa.binary()),
            pa.field("fixed", pa.binary(4)),
            pa.field("price", pa.decimal128(9, 2)),
            pa.field("day", pa.date32()),
            pa.field("clock", pa.time64("us")),
            pa.field("moment", pa.timestamp("ms", tz="UTC")),
            pa.field("local", pa.timestamp("us")),
        ]
    )
    table = pa.table(
        {
            "flag": [True, False, True],
            "small": [1, None, -3],
            "count": [10, 20, None],
            "total": [100, 200, 300],
            "unsigned": [1, 2, 3],
            "ratio": [1.5, 2.5, None],
            "weight": [1.25, 2.5, 3.75],
            "name": ["GB", "JP", None],
            "blob": [b"one", b"two", b"three"],
            "fixed": [b"abcd", b"efgh", b"ijkl"],
            "price": [decimal.Decimal("1.25"), decimal.Decimal("2.50"), None],
            "day": [datetime.date(2026, 8, 28)] * 3,
            "clock": [datetime.time(12, 30, 15)] * 3,
            "moment": [datetime.datetime(2026, 8, 28, 12, 0, tzinfo=datetime.UTC)] * 3,
            "local": [datetime.datetime(2026, 8, 28, 12, 0)] * 3,
        },
        schema=schema,
    )
    pq.write_table(
        table,
        "alltypes.parquet",
        compression="none",
        version="2.6",
        write_statistics=True,
        store_schema=True,
    )


def plain():
    """The same columns as alltypes with nothing encoded, so a decoder of plain
    values has one of each to read.

    alltypes is what pyarrow writes by default, which means a dictionary in
    front of nearly every column, and a dictionary page says nothing about how a
    plain value is read. This is the other half: no dictionary, no compression,
    no statistics, and one page per column holding the values as they are.

    The decimal is here for the column that cannot be read yet, so that the
    refusal has a real file behind it rather than a made up one.
    """
    schema = pa.schema(
        [
            pa.field("flag", pa.bool_()),
            pa.field("small", pa.int8()),
            pa.field("short", pa.int16()),
            pa.field("count", pa.int32()),
            pa.field("total", pa.int64()),
            pa.field("byte", pa.uint8()),
            pa.field("ushort", pa.uint16()),
            pa.field("unsigned", pa.uint32()),
            pa.field("big", pa.uint64()),
            pa.field("ratio", pa.float32()),
            pa.field("weight", pa.float64()),
            pa.field("name", pa.string()),
            pa.field("blob", pa.binary()),
            pa.field("fixed", pa.binary(4)),
            pa.field("price", pa.decimal128(9, 2)),
            pa.field("day", pa.date32()),
            pa.field("clock", pa.time64("us")),
            pa.field("moment", pa.timestamp("ms", tz="UTC")),
            pa.field(
                "point",
                pa.struct(
                    [pa.field("x", pa.float64(), nullable=False), pa.field("y", pa.float64())]
                ),
            ),
        ]
    )
    table = pa.table(
        {
            "flag": [True, False, None, True],
            "small": [1, -2, None, 127],
            "short": [1000, -2000, None, 32767],
            "count": [10, -20, None, 30],
            "total": [100, -200, None, 300],
            "byte": [1, 2, None, 255],
            "ushort": [1, 2, None, 65535],
            "unsigned": [1, 2, None, 4294967295],
            "big": [1, 2, None, 18446744073709551615],
            "ratio": [1.5, -2.5, None, 3.5],
            "weight": [1.25, -2.5, None, 3.75],
            "name": ["one", "", None, "four"],
            "blob": [b"one", b"", None, b"four"],
            "fixed": [b"abcd", b"efgh", None, b"mnop"],
            "price": [decimal.Decimal("1.25"), decimal.Decimal("-2.50"), None, decimal.Decimal("3.75")],
            "day": [datetime.date(2026, 8, 28), datetime.date(1969, 7, 20), None, datetime.date(1970, 1, 1)],
            "clock": [datetime.time(12, 30, 15), datetime.time(0, 0, 0), None, datetime.time(23, 59, 59)],
            "moment": [
                datetime.datetime(2026, 8, 28, 12, 0, tzinfo=datetime.UTC),
                datetime.datetime(1969, 7, 20, 20, 17, 40, tzinfo=datetime.UTC),
                None,
                datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC),
            ],
            # A struct with an optional field in it, which is the flattest
            # column that needs a level of more than one bit. point.y is two
            # deep, so a missing y and a missing point are told apart in the
            # levels and are the same null in a flat column.
            "point": [
                {"x": 1.0, "y": 2.0},
                {"x": 3.0, "y": None},
                None,
                {"x": 5.0, "y": 6.0},
            ],
        },
        schema=schema,
    )
    pq.write_table(
        table,
        "plain.parquet",
        compression="none",
        version="2.6",
        use_dictionary=False,
        write_statistics=False,
        store_schema=False,
    )


def chunks():
    """Two row groups of one column, compressed, so that a reader sees both."""
    table = pa.table({"code": ["GB", "JP", "US", "FR", "DE", "GB"], "n": list(range(6))})
    pq.write_table(
        table,
        "chunks.parquet",
        compression="snappy",
        row_group_size=3,
        version="2.6",
        write_statistics=True,
        store_schema=False,
    )


def nested():
    """Lists, maps and structs, which are what a schema tree is for.

    Written without the Arrow schema, so that the parquet schema is all there is
    to work from and the three level lists and the key_value groups have to be
    read the way the format describes them.
    """
    schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("counts", pa.list_(pa.field("item", pa.int32(), nullable=False))),
            pa.field("props", pa.map_(pa.string(), pa.int64())),
            pa.field(
                "point",
                pa.struct(
                    [
                        pa.field("x", pa.float64(), nullable=False),
                        pa.field("y", pa.float64()),
                    ]
                ),
            ),
            pa.field("matrix", pa.list_(pa.list_(pa.int32()))),
            pa.field(
                "people",
                pa.list_(pa.struct([pa.field("name", pa.string()), pa.field("age", pa.int32())])),
            ),
        ]
    )
    table = pa.table(
        {
            "id": [1, 2],
            "tags": [["north", "east"], None],
            "counts": [[1, 2, 3], []],
            "props": [[("width", 10), ("height", 20)], None],
            "point": [{"x": 1.5, "y": 2.5}, {"x": 3.5, "y": None}],
            "matrix": [[[1], [2, 3]], None],
            "people": [[{"name": "ann", "age": 30}], []],
        },
        schema=schema,
    )
    pq.write_table(
        table,
        "nested.parquet",
        compression="none",
        version="2.6",
        write_statistics=True,
        store_schema=False,
    )


def pages():
    """Several pages per column, written the second way, with checksums.

    The other files here are one page per column written the first way, which is
    what pyarrow does unless it is told otherwise, so this is the other half of
    what a page walker has to read. The batch size is small so that a column
    ends up with more than one page in it without the file getting big, and the
    string column is dictionary encoded so that there is a dictionary page in
    front of its data pages.

    The nullable column is here for the null count, which the second version of
    the data page writes down and the first one does not.
    """
    rows = 500
    words = ["alpha", "beta", "gamma", "delta"]
    table = pa.table(
        {
            "n": pa.array(range(rows), pa.int32()),
            "word": pa.array([words[i % len(words)] for i in range(rows)], pa.string()),
            "maybe": pa.array([None if i % 3 == 0 else i for i in range(rows)], pa.int32()),
        }
    )
    pq.write_table(
        table,
        "pages.parquet",
        compression="none",
        version="2.6",
        data_page_version="2.0",
        data_page_size=1024,
        write_batch_size=100,
        write_page_checksum=True,
        use_dictionary=["word"],
        store_schema=False,
    )


def dictionary():
    """A column of more distinct values than a few bits of index can hold.

    pages.parquet already has a dictionary in front of a column of four words,
    which is two bit indices, and alltypes has columns of one distinct value,
    which is nought bit indices and no bytes at all. This is the other end.
    Three hundred distinct values needs nine bits, so the indices are packed
    across byte boundaries and a value is read out of two of them, and there
    are enough rows for several pages of them.

    The nullable column is the two halves together, since the indices are only
    written for the rows that have a value and the levels are the only thing
    that says which those are.
    """
    rows = 1000
    table = pa.table(
        {
            "code": pa.array(["c%03d" % (i % 300) for i in range(rows)], pa.string()),
            "size": pa.array([None if i % 7 == 0 else i % 300 for i in range(rows)], pa.int64()),
        }
    )
    pq.write_table(
        table,
        "dictionary.parquet",
        compression="none",
        version="2.6",
        data_page_size=1024,
        write_batch_size=100,
        write_statistics=False,
        store_schema=False,
    )


def fallback():
    """A chunk that starts as indices into a dictionary and gives up half way.

    A writer keeps a dictionary while the distinct values fit in one page of
    them and writes plain pages for the rest of the chunk once they do not, so
    the chunk is indices at the front and values at the back. Every value here
    is distinct and the limit is small, so the writer runs out early.
    """
    rows = 2000
    table = pa.table({"code": pa.array(["value-%06d" % i for i in range(rows)], pa.string())})
    pq.write_table(
        table,
        "fallback.parquet",
        compression="none",
        version="2.6",
        dictionary_pagesize_limit=1024,
        data_page_size=1024,
        write_statistics=False,
        store_schema=False,
    )


def delta():
    """Integers written as differences rather than as values.

    A column of row numbers, or of timestamps a second apart, is a column whose
    differences are all the same, and writing the differences costs a bit a row
    where writing the values costs thirty two. That is what this encoding is
    for, and the columns here are the shapes it has to read: a run of a constant
    difference, a run that goes down instead of up, differences that need most
    of an int64 to hold, and a column with holes in it where the differences are
    between the values that are there rather than between the rows.

    The narrow columns are here because parquet has no type for them and writes
    them as int32, so they arrive as differences of int32 and have to be
    narrowed like any other value. The unsigned one runs past where an int32
    turns negative, which is where the arithmetic has to wrap the way the writer
    wrapped it.

    The page size is small so that a column ends up in several pages, since the
    encoding starts again from a first value in each of them.
    """
    rows = 1000
    table = pa.table(
        {
            "n": pa.array(range(rows), pa.int32()),
            "small": pa.array([(i % 251) - 125 for i in range(rows)], pa.int8()),
            "unsigned": pa.array([(i * 4000000) % (1 << 32) for i in range(rows)], pa.uint32()),
            "down": pa.array([rows - i for i in range(rows)], pa.int32()),
            "wobble": pa.array([(i % 17) * (1000 if i % 2 else -1000) for i in range(rows)], pa.int32()),
            "big": pa.array(
                [((i * 6364136223846793005 + 1442695040888963407) % (1 << 62)) for i in range(rows)],
                pa.int64(),
            ),
            "maybe": pa.array([None if i % 5 == 0 else i * 7 for i in range(rows)], pa.int64()),
        }
    )
    pq.write_table(
        table,
        "delta.parquet",
        compression="none",
        version="2.6",
        use_dictionary=False,
        column_encoding={name: "DELTA_BINARY_PACKED" for name in table.column_names},
        data_page_size=1024,
        write_batch_size=100,
        write_statistics=False,
        store_schema=False,
    )


def strings():
    """Byte arrays written as differences rather than as values.

    There are two of these encodings and this file has both. The first writes
    the lengths of the values as differences and then all the bytes end to end,
    which saves the four bytes of length that sit in front of every value of a
    plain page. The second writes how much of each value the one before it
    already said, then the rest of it, which is what turns a sorted column of
    paths or keys into a few bytes a row.

    The columns are the shapes the two have to read. A run of keys with a long
    shared prefix is what the second one is for, and a column of one repeated
    value is the far end of it, where the prefix is the whole value and the rest
    of it is nothing at all. The blobs go the other way and share nothing, and
    have empty values scattered through them, which are the values a length of
    nought has to produce rather than a null. The fixed width column is here
    because the format allows the second encoding for it and not the first, so
    it is the one column where the two are told apart by the schema.

    The page size is small so that a column ends up in several pages, since both
    encodings start again from nothing in each of them.
    """
    rows = 1000
    table = pa.table(
        {
            "key": pa.array([f"customer/2026/08/{i:06d}" for i in range(rows)]),
            "word": pa.array([("a" * (i % 17)) + str(i % 97) for i in range(rows)]),
            "blob": pa.array(
                [b"" if i % 13 == 0 else bytes([i % 256]) * (i % 11) for i in range(rows)],
                pa.binary(),
            ),
            "maybe": pa.array([None if i % 7 == 0 else f"row-{i}" for i in range(rows)]),
            "same": pa.array(["the same string every time"] * rows),
            "fixed": pa.array([f"{i:08d}".encode() for i in range(rows)], pa.binary(8)),
        }
    )
    pq.write_table(
        table,
        "strings.parquet",
        compression="none",
        version="2.6",
        use_dictionary=False,
        column_encoding={
            "key": "DELTA_BYTE_ARRAY",
            "word": "DELTA_LENGTH_BYTE_ARRAY",
            "blob": "DELTA_LENGTH_BYTE_ARRAY",
            "maybe": "DELTA_BYTE_ARRAY",
            "same": "DELTA_BYTE_ARRAY",
            "fixed": "DELTA_BYTE_ARRAY",
        },
        data_page_size=1024,
        write_batch_size=100,
        write_statistics=False,
        store_schema=False,
    )


def codecs():
    """The same rows written with a different codec per column, twice over.

    Compression is per column chunk rather than per file, which is easy to say
    and easy to write a reader that quietly assumes otherwise, so every column
    here is compressed differently and one of them is not compressed at all.
    The brotli one is there to be refused by name: this package does not have
    that codec and a file holding one still has to be readable up to it.

    Two files because the two versions of the data page put their levels in
    different places. The first version compresses the levels along with the
    values, so the whole body goes through the codec. The second leaves the
    levels alone in front of the compressed values, which means the page comes
    back as two pieces that have to be put together, and that is worth a file of
    its own.

    The page size is small so that a column ends up in several pages, since a
    codec is undone once per page and a chunk of one page proves less.
    """
    rows = 1000
    words = ["alpha", "beta", "gamma", "delta"]
    table = pa.table(
        {
            "plain": pa.array([f"row-{i}" for i in range(rows)], pa.string()),
            "zip": pa.array([i * 3 for i in range(rows)], pa.int64()),
            "snap": pa.array([f"customer/2026/08/{i:06d}" for i in range(rows)], pa.string()),
            "word": pa.array([words[i % len(words)] for i in range(rows)], pa.string()),
            "maybe": pa.array([None if i % 5 == 0 else i for i in range(rows)], pa.int32()),
            "br": pa.array([float(i) / 3 for i in range(rows)], pa.float64()),
        }
    )
    compression = {
        "plain": "none",
        "zip": "gzip",
        "snap": "snappy",
        "word": "snappy",
        "maybe": "gzip",
        "br": "brotli",
    }
    for name, version in [("codecs.parquet", "1.0"), ("codecs2.parquet", "2.0")]:
        pq.write_table(
            table,
            name,
            compression=compression,
            version="2.6",
            data_page_version=version,
            data_page_size=1024,
            write_batch_size=100,
            use_dictionary=["word"],
            write_statistics=False,
            store_schema=False,
        )


def legacy():
    """Timestamps as int96, which is how parquet wrote them before it had a type.

    Nothing has written one for years and everybody still has to read them. The
    value is twelve bytes, a count of nanoseconds into the day and then the
    Julian day it is in, which is a day number counted from a morning in 4713
    BC. One of the two rows is before 1970 so that the day comes out negative
    once it is moved to the epoch everything else counts from.

    Written the old way all round, so the schema has converted types and no
    logical ones and the pages are the first version of the data page.
    """
    schema = pa.schema([pa.field("moment", pa.timestamp("ns")), pa.field("label", pa.string())])
    table = pa.table(
        {
            "moment": [
                datetime.datetime(2026, 8, 28, 12, 0, 0, 123456),
                datetime.datetime(1969, 7, 20, 20, 17, 40),
            ],
            "label": ["now", "then"],
        },
        schema=schema,
    )
    pq.write_table(
        table,
        "legacy.parquet",
        compression="none",
        version="1.0",
        use_deprecated_int96_timestamps=True,
        use_dictionary=False,
        write_statistics=False,
        store_schema=False,
    )


def stats():
    """Row groups a scan can tell apart without reading them.

    Three groups of four rows with n running from 0 to 11, so every group holds
    a range of its own and a filter on n has something to decide with. The rest
    of the columns are the ones whose bounds are easy to read wrongly: an
    unsigned column holding a value that comes out negative when it is read
    signed, a float column with a NaN in it, a string column whose smallest
    value is the empty string, and a column that is missing everywhere and has
    no bounds at all.

    Written without a dictionary so that the pages are the values themselves,
    and with the statistics on, which is what the file is for.
    """
    schema = pa.schema(
        [
            pa.field("n", pa.int64(), nullable=False),
            pa.field("word", pa.string(), nullable=False),
            pa.field("size", pa.uint32(), nullable=False),
            pa.field("ratio", pa.float64(), nullable=False),
            pa.field("absent", pa.int32()),
            pa.field("flag", pa.bool_(), nullable=False),
        ]
    )
    table = pa.table(
        {
            "n": list(range(12)),
            "word": ["", "delta", "alpha", "echo"]
            + ["mike", "november", "oscar", "papa"]
            + ["zulu", "yankee", "victor", "sierra"],
            "size": [1, 4294967295, 7, 9] + [10, 11, 12, 13] + [20, 21, 22, 23],
            "ratio": [1.5, 2.5, 0.5, 3.5]
            + [1.0, float("nan"), 2.0, 3.0]
            + [-1.0, 0.0, 1.0, 2.0],
            "absent": [None] * 12,
            "flag": [True, False, True, False] * 3,
        },
        schema=schema,
    )
    pq.write_table(
        table,
        "stats.parquet",
        compression="none",
        row_group_size=4,
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
        store_schema=False,
    )


def index():
    """Pages a scan can tell apart without reading them.

    Two row groups of two hundred rows and a batch size that puts a hundred rows
    in every page, so each group holds two pages of each column and the page
    boundaries are where they can be checked against. The columns are written to
    give the boundary order all three of its values: one runs up, one runs down,
    and one holds a page whose values are on both sides of the page before it,
    which is neither. The nullable one is missing for the whole of the second
    page of each group, which is how a page the index calls null gets written at
    all, and it is the reason the null counts are there.

    Written with the page index on, which pyarrow does not do unless it is
    asked, and with the second version of the data page so that the null counts
    in the index can be checked against the ones in the page headers.
    """
    rows = 400
    schema = pa.schema(
        [
            pa.field("n", pa.int32(), nullable=False),
            pa.field("down", pa.int32(), nullable=False),
            pa.field("wave", pa.int32(), nullable=False),
            pa.field("word", pa.string(), nullable=False),
            pa.field("gap", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "n": list(range(rows)),
            "down": list(range(rows - 1, -1, -1)),
            "wave": [10 + i % 11 if i // 100 % 2 == 0 else i % 31 for i in range(rows)],
            "word": ["w%04d" % i for i in range(rows)],
            "gap": [None if i % 200 >= 100 else i for i in range(rows)],
        },
        schema=schema,
    )
    pq.write_table(
        table,
        "index.parquet",
        compression="none",
        row_group_size=200,
        version="2.6",
        data_page_version="2.0",
        data_page_size=64,
        write_batch_size=100,
        use_dictionary=False,
        write_statistics=True,
        write_page_index=True,
        store_schema=False,
    )


def empty():
    """A file with a schema and no rows, which still has a footer."""
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("label", pa.string())])
    pq.write_table(schema.empty_table(), "empty.parquet", compression="none", store_schema=False)


if __name__ == "__main__":
    alltypes()
    plain()
    chunks()
    nested()
    pages()
    dictionary()
    fallback()
    delta()
    strings()
    codecs()
    legacy()
    stats()
    index()
    empty()
    print(
        "wrote alltypes.parquet, plain.parquet, chunks.parquet, nested.parquet, "
        "pages.parquet, dictionary.parquet, fallback.parquet, delta.parquet, "
        "strings.parquet, codecs.parquet, codecs2.parquet, legacy.parquet, "
        "stats.parquet, index.parquet and empty.parquet"
    )
