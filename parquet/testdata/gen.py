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


def empty():
    """A file with a schema and no rows, which still has a footer."""
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("label", pa.string())])
    pq.write_table(schema.empty_table(), "empty.parquet", compression="none", store_schema=False)


if __name__ == "__main__":
    alltypes()
    chunks()
    nested()
    pages()
    legacy()
    empty()
    print(
        "wrote alltypes.parquet, chunks.parquet, nested.parquet, pages.parquet, "
        "legacy.parquet and empty.parquet"
    )
