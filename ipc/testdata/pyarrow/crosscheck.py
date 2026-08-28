"""Check kuma against pyarrow across the Arrow C data interface.

This loads the shared library built from crosscheck.go next to it and runs
every case in both directions. It is driven by TestPyarrow, and takes the path
to the library as its only argument.

Nothing here knows anything about kuma beyond the seven functions the library
exports. That is the point: pyarrow is a second implementation of the same
specification, written by somebody else, so what the two of them agree on is
the specification rather than kuma's reading of it.
"""

import ctypes
import sys
from decimal import Decimal

import pyarrow as pa


class ArrowSchema(ctypes.Structure):
    pass


ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class ArrowArray(ctypes.Structure):
    pass


ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]

SCHEMA = ctypes.POINTER(ArrowSchema)
ARRAY = ctypes.POINTER(ArrowArray)


def load(path):
    lib = ctypes.CDLL(path)
    lib.kuma_case_count.restype = ctypes.c_int
    lib.kuma_case_count.argtypes = []
    for name in ("kuma_case_name", "kuma_case_kind"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_char_p
        fn.argtypes = [ctypes.c_int]
    lib.kuma_case_export.restype = ctypes.c_char_p
    lib.kuma_case_export.argtypes = [ctypes.c_int, SCHEMA, ARRAY]
    lib.kuma_case_verify.restype = ctypes.c_char_p
    lib.kuma_case_verify.argtypes = [ctypes.c_int, SCHEMA, ARRAY]
    lib.kuma_pass_through.restype = ctypes.c_char_p
    lib.kuma_pass_through.argtypes = [SCHEMA, ARRAY, SCHEMA, ARRAY]
    lib.kuma_release_live.restype = None
    lib.kuma_release_live.argtypes = []
    return lib


def addresses(array):
    """The address of every buffer of a pyarrow array.

    A missing validity bitmap comes back as None, and the C interface writes
    that as a null pointer, so both of them are zero here.
    """
    return [0 if b is None else b.address for b in array.buffers()]


def struct_addresses(values):
    """The address of every buffer of a C array struct, as written."""
    return [values.buffers[i] or 0 for i in range(values.n_buffers)]


def shared(got, want, where):
    """Report the buffers of two arrays that are not the same memory.

    Only the buffers both sides have are compared. kuma writes one buffer the
    other side does not, the sizes of the data blocks, and a column of text
    that came in as offsets gets a buffer of views that had to be built, so the
    counts are not always equal and only the ones that line up mean anything.

    A buffer of no bytes is a null pointer on one side and an allocation of
    nothing on the other, and neither of them can be a copy of anything, so a
    pair where either address is zero is left alone.
    """
    problems = []
    for i in range(min(len(got), len(want))):
        if got[i] and want[i] and got[i] != want[i]:
            problems.append(
                "%s buffer %d is at %x, want %x" % (where, i, got[i], want[i])
            )
    return problems


def go_origin(lib):
    """kuma exports, pyarrow imports and hands it straight back.

    The type is checked against the name pyarrow prints for it, the buffers
    against the addresses kuma wrote, and the values by kuma itself once the
    array has been through pyarrow and returned.
    """
    problems = []
    for i in range(lib.kuma_case_count()):
        name = lib.kuma_case_name(i).decode()
        kind = lib.kuma_case_kind(i).decode()

        schema, values = ArrowSchema(), ArrowArray()
        err = lib.kuma_case_export(i, ctypes.byref(schema), ctypes.byref(values))
        if err is not None:
            problems.append("%s: export: %s" % (name, err.decode()))
            continue

        exported = struct_addresses(values)
        array = pa.Array._import_from_c(
            ctypes.addressof(values), ctypes.addressof(schema)
        )
        if str(array.type) != kind:
            problems.append("%s: pyarrow reads the type as %s, want %s"
                            % (name, array.type, kind))
            continue
        problems += shared(addresses(array), exported, "%s:" % name)

        back_schema, back_values = ArrowSchema(), ArrowArray()
        array._export_to_c(
            ctypes.addressof(back_values), ctypes.addressof(back_schema)
        )
        err = lib.kuma_case_verify(
            i, ctypes.byref(back_schema), ctypes.byref(back_values)
        )
        if err is not None:
            problems.append("%s: back from pyarrow: %s" % (name, err.decode()))
    return problems


# The cases that start in pyarrow. Each is a name, an array, the type kuma
# hands back, and whether the buffers that come back should be the ones that
# went in.
#
# kuma has one layout for text rather than three, so a column of strings that
# arrives as offsets leaves as views. The characters are still the same
# characters in the same place, which is why those cases still check that the
# data buffer is shared, but the array is not the array it was and pyarrow is
# right to say so.
def python_cases():
    long = "a string too long to live inside a view"
    return [
        ("int64", pa.array([1, -2, 3], pa.int64()), "int64", True),
        ("int32", pa.array([1, -2, 3], pa.int32()), "int32", True),
        ("uint8", pa.array([1, 2, 3], pa.uint8()), "uint8", True),
        ("float64", pa.array([1.5, -2.5], pa.float64()), "double", True),
        ("float32", pa.array([1.5, -2.5], pa.float32()), "float", True),
        ("bool", pa.array([True, False, True], pa.bool_()), "bool", True),
        ("nulls", pa.array([1, None, 3], pa.int64()), "int64", True),
        ("null type", pa.nulls(4), "null", False),
        ("timestamp", pa.array([1, 2], pa.timestamp("us", tz="UTC")),
         "timestamp[us, tz=UTC]", True),
        ("date32", pa.array([1, 2], pa.date32()), "date32[day]", True),
        ("duration", pa.array([1, 2], pa.duration("s")), "duration[s]", True),
        ("decimal128", pa.array([Decimal("1.25"), Decimal("2.50")],
         pa.decimal128(18, 2)),
         "decimal128(18, 2)", True),
        ("fixed size binary", pa.array([b"abcd", b"efgh"],
         pa.binary(4)), "fixed_size_binary[4]", True),
        ("string", pa.array(["a", long, "bb"], pa.string()),
         "string_view", False),
        ("large string", pa.array(["a", long, "bb"], pa.large_string()),
         "string_view", False),
        ("binary", pa.array([b"a", long.encode(), b"bb"], pa.binary()),
         "binary_view", False),
        ("large binary", pa.array([b"a", long.encode()], pa.large_binary()),
         "binary_view", False),
        ("string view", pa.array(["a", long, "bb"], pa.string_view()),
         "string_view", True),
        ("binary view", pa.array([b"a", long.encode()], pa.binary_view()),
         "binary_view", True),
        ("sliced", pa.array([10, 20, 30, 40, 50], pa.int64()).slice(1, 3),
         "int64", True),
        ("empty", pa.array([], pa.int64()), "int64", True),
    ]


def python_origin(lib):
    """pyarrow exports, kuma imports and exports it again without a copy."""
    problems = []
    for name, array, kind, same in python_cases():
        before = addresses(array)

        schema, values = ArrowSchema(), ArrowArray()
        array._export_to_c(ctypes.addressof(values), ctypes.addressof(schema))

        back_schema, back_values = ArrowSchema(), ArrowArray()
        err = lib.kuma_pass_through(
            ctypes.byref(schema), ctypes.byref(values),
            ctypes.byref(back_schema), ctypes.byref(back_values),
        )
        if err is not None:
            problems.append("%s: pass through: %s" % (name, err.decode()))
            continue

        back = pa.Array._import_from_c(
            ctypes.addressof(back_values), ctypes.addressof(back_schema)
        )
        if str(back.type) != kind:
            problems.append("%s: came back as %s, want %s"
                            % (name, back.type, kind))
        elif back.to_pylist() != array.to_pylist():
            problems.append("%s: came back as %s, want %s"
                            % (name, back.to_pylist(), array.to_pylist()))
        if same:
            problems += shared(addresses(back), before, "%s:" % name)
        else:
            problems += data_shared(name, back, array)

        back = None
        lib.kuma_release_live()
    return problems


def data_shared(name, back, array):
    """Check that the characters of a column of text were not copied.

    A column that arrived as offsets leaves as views over the same bytes, so
    the buffer of views is new and the block the views point into is the buffer
    the characters arrived in. A column with nothing long enough to need a
    block has no block, and there is nothing to check.
    """
    if array.type == pa.null():
        return []
    got, want = addresses(back), addresses(array)
    if len(got) < 3:
        return []
    if got[2] != want[2]:
        return ["%s: the characters are at %x, want %x"
                % (name, got[2], want[2])]
    return []


def main():
    if len(sys.argv) != 2:
        print("usage: crosscheck.py LIBRARY", file=sys.stderr)
        return 2

    lib = load(sys.argv[1])
    problems = go_origin(lib) + python_origin(lib)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("pyarrow %s, %d cases out of kuma, %d cases into it"
          % (pa.__version__, lib.kuma_case_count(), len(python_cases())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
