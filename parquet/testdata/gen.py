"""Write the parquet files the footer tests read.

The files are checked in rather than written by the tests, because the tests run
on every platform in the matrix and pyarrow is only installed on one of them. A
reader has to be checked against files somebody else wrote, so these are written
by pyarrow and read by kuma, and this script is here so that anybody can see
what produced them and write them again.

Run it from this directory with pyarrow installed:

    python3 gen.py

Keep the files small. They are read for their footers and nobody is measuring
anything with them.
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


def empty():
    """A file with a schema and no rows, which still has a footer."""
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("label", pa.string())])
    pq.write_table(schema.empty_table(), "empty.parquet", compression="none", store_schema=False)


if __name__ == "__main__":
    alltypes()
    chunks()
    empty()
    print("wrote alltypes.parquet, chunks.parquet and empty.parquet")
