package kernel

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Groups is rows divided up by the values of one or more key columns.
//
// It is what [GroupBy] works out and what every aggregation is handed. Working
// it out once and passing it around is the point: a query that asks for a sum,
// a mean and a count over the same keys divides the rows up once and then makes
// three cheap passes rather than three expensive ones.
//
// The groups are numbered in the order they first appear in the rows, which is
// deterministic without being sorted. A caller who wants them sorted sorts the
// result, and pays for the sort only when they want it. The pandas default is
// to sort, and turning it off is the single most common thing people do to a
// group by there.
type Groups struct {
	// ids[i] is the group row i belongs to.
	ids []int

	// first[g] is the first row of group g, which is also what fixes the
	// numbering.
	first []int

	// sizes[g] is how many rows are in group g, nulls included.
	sizes []int

	// keys holds one row per group, the distinct key values, in group order.
	keys []*array.Chunked
}

// Len returns how many rows were grouped.
func (g *Groups) Len() int { return len(g.ids) }

// NumGroups returns how many distinct keys there are.
func (g *Groups) NumGroups() int { return len(g.first) }

// IDs returns the group of every row, in row order. The caller must not modify
// the result.
func (g *Groups) IDs() []int { return g.ids }

// FirstRows returns the first row of every group, in group order. The caller
// must not modify the result.
//
// This is the positional first that SQL calls ANY_VALUE and pandas calls
// nth(0), and it is not what [First] returns, since that one skips over the
// missing values. Gathering a column at these positions is how to get it.
func (g *Groups) FirstRows() []int { return g.first }

// Sizes returns how many rows are in every group, in group order, counting the
// rows whose values are missing. The caller must not modify the result.
func (g *Groups) Sizes() []int { return g.sizes }

// Keys returns the distinct key values, one row per group, in group order.
//
// There is one column here for every column [GroupBy] was given, in the order
// they were given. The caller must not modify the result.
func (g *Groups) Keys() []*array.Chunked { return g.keys }

// GroupBy divides rows up by the values of the key columns.
//
// Two rows are in the same group when every key agrees, and a missing value
// agrees with a missing value. That is what SQL does and what Polars does. The
// pandas default is to drop the rows whose key is missing, which loses data
// quietly and is the wrong thing for a library people will use to count
// things.
//
// NaN keys land in one group rather than in one group each. There are millions
// of bit patterns that are NaN and telling them apart would put a row in a
// group of its own for a reason nobody asked about. The same goes for negative
// zero, which groups with zero because it is equal to it.
//
// It reports an error if a key column holds values that cannot be compared for
// equality, which today means the nested types.
//
// It panics if there are no keys, if a key is nil, or if the keys are not all
// the same length, since all three are a mistake in the program rather than
// something the data did.
func GroupBy(keys ...*array.Chunked) (*Groups, error) {
	if len(keys) == 0 {
		panic("kernel: group by with no keys")
	}

	n := -1
	ks := make([]*key, len(keys))
	for i, c := range keys {
		if c == nil {
			panic(fmt.Sprintf("kernel: group by key %d is a nil column", i))
		}
		if n < 0 {
			n = c.Len()
		} else if c.Len() != n {
			panic(fmt.Sprintf("kernel: group by key %d has %d rows, key 0 has %d",
				i, c.Len(), n))
		}
		k, err := newKey(c)
		if err != nil {
			return nil, err
		}
		ks[i] = k
	}

	g := &Groups{ids: make([]int, n)}

	// The map is keyed by the encoded bytes of the whole key. Looking a byte
	// slice up in a map keyed by strings does not copy it, and only a key that
	// turns out to be new is turned into a string that lasts, so the cost of a
	// row that belongs to a group already found is a hash and a compare.
	seen := make(map[string]int)
	var scratch []byte
	for i := range n {
		scratch = scratch[:0]
		for _, k := range ks {
			scratch = k.appendRow(scratch, i)
		}

		id, ok := seen[string(scratch)]
		if !ok {
			id = len(g.first)
			seen[string(scratch)] = id
			g.first = append(g.first, i)
			g.sizes = append(g.sizes, 0)
		}
		g.ids[i] = id
		g.sizes[id]++
	}

	g.keys = make([]*array.Chunked, len(keys))
	for i, c := range keys {
		g.keys[i] = Take(c, g.first)
	}
	return g, nil
}

// OneGroup returns n rows in a single group.
//
// This is what makes an aggregation over a whole column and an aggregation over
// a group the same piece of code. Sum of a series is Sum of one group, and
// there is no second implementation to keep in step with the first.
func OneGroup(n int) *Groups {
	if n < 0 {
		panic(fmt.Sprintf("kernel: one group of %d rows", n))
	}

	g := &Groups{ids: make([]int, n)}
	if n > 0 {
		g.first = []int{0}
		g.sizes = []int{n}
	}
	return g
}

// key is one key column of a group by, positioned at a row.
//
// The rows are visited in order, once each, so the chunk a row is in is found
// by walking forward from the last one rather than by searching. The typed view
// of the values is taken once per chunk for the same reason: a group by reads
// every row of every key exactly once, and the work per row should be the read
// and nothing else.
type key struct {
	chunks []*array.Array

	// chunk is the chunk the last row asked for was in, and base is the row
	// that chunk begins at.
	chunk int
	base  int

	// bind returns a writer over one chunk, and write is the one bound to the
	// chunk above.
	bind  func(*array.Array) writer
	write writer
}

// writer appends the bytes of value i of the chunk it was bound to.
type writer func(dst []byte, i int) []byte

// appendRow appends the encoded key of row i to dst. Rows have to be asked for
// in order.
func (k *key) appendRow(dst []byte, i int) []byte {
	for i >= k.base+k.chunks[k.chunk].Len() {
		k.base += k.chunks[k.chunk].Len()
		k.chunk++
		k.write = k.bind(k.chunks[k.chunk])
	}

	a, j := k.chunks[k.chunk], i-k.base
	if a.IsNull(j) {
		// A byte that no value can start with, so a missing value groups with
		// the other missing values and with nothing else.
		return append(dst, 0)
	}
	return k.write(append(dst, 1), j)
}

// newKey returns the key over c, or an error if its values cannot be compared
// for equality.
func newKey(c *array.Chunked) (*key, error) {
	bind, err := binderFor(c.DType())
	if err != nil {
		return nil, err
	}

	k := &key{chunks: c.Chunks(), bind: bind}
	if len(k.chunks) == 0 {
		// An empty column is never asked for a row, so there is nothing to bind
		// to and nothing that will ask.
		k.chunks = []*array.Array{array.NewNull(0)}
		k.write = func(dst []byte, _ int) []byte { return dst }
		return k, nil
	}
	k.write = bind(k.chunks[0])
	return k, nil
}

// binderFor returns the way values of a type are written into a key.
//
// Every encoding here has to be injective: two values that are equal have to
// produce the same bytes and two that are not have to produce different ones.
// That is a weaker thing to ask for than an order, which is why the decimals
// are here and are not in the sort. Two decimals of one column have the same
// precision and scale, so equal values are equal bytes, and it does not matter
// that the byte order says nothing about which is bigger.
func binderFor(dt dtype.DataType) (func(*array.Array) writer, error) {
	switch dt.Kind() {
	case dtype.NullKind:
		// Every value is missing, so appendRow answers before it gets here.
		return func(*array.Array) writer {
			return func(dst []byte, _ int) []byte { return dst }
		}, nil
	case dtype.BoolKind:
		return bindBools, nil
	case dtype.Int8Kind:
		return bindSigned[int8], nil
	case dtype.Int16Kind:
		return bindSigned[int16], nil
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return bindSigned[int32], nil
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return bindSigned[int64], nil
	case dtype.Uint8Kind:
		return bindUnsigned[uint8], nil
	case dtype.Uint16Kind:
		return bindUnsigned[uint16], nil
	case dtype.Uint32Kind:
		return bindUnsigned[uint32], nil
	case dtype.Uint64Kind:
		return bindUnsigned[uint64], nil
	case dtype.Float32Kind:
		return bindFloats[float32], nil
	case dtype.Float64Kind:
		return bindFloats[float64], nil
	case dtype.StringKind, dtype.BinaryKind, dtype.LargeStringKind,
		dtype.LargeBinaryKind, dtype.FixedSizeBinaryKind,
		dtype.Decimal128Kind, dtype.Decimal256Kind, dtype.IntervalKind:
		return bindBytes, nil
	default:
		// A list or a struct is equal to another one when its parts are, which
		// is a walk over the children rather than a read of a value, and it
		// wants writing rather than guessing at.
		return nil, fmt.Errorf("kernel: a %s column cannot be a group by key yet", dt)
	}
}

// bindSigned writes an integer widened to eight bytes. Widening a signed value
// keeps it apart from every other value of its own column, which is all a key
// has to do, and it means one encoding covers the four widths and the dates and
// the timestamps as well.
func bindSigned[T array.Numeric](a *array.Array) writer {
	vs := a.Values[T]()
	return func(dst []byte, i int) []byte {
		return binary.LittleEndian.AppendUint64(dst, uint64(int64(vs[i])))
	}
}

// bindUnsigned is bindSigned without the sign extension.
func bindUnsigned[T array.Numeric](a *array.Array) writer {
	vs := a.Values[T]()
	return func(dst []byte, i int) []byte {
		return binary.LittleEndian.AppendUint64(dst, uint64(vs[i]))
	}
}

// bindFloats writes a float widened to eight bytes, with the two values that
// have more than one bit pattern given the one they group under.
//
// Widening a float32 to a float64 is exact, so two float32 values that differ
// still differ afterwards.
func bindFloats[T float32 | float64](a *array.Array) writer {
	vs := a.Values[T]()
	return func(dst []byte, i int) []byte {
		v := float64(vs[i])
		switch {
		case math.IsNaN(v):
			// Every NaN groups with every other NaN. Millions of bit patterns
			// are NaN and a group per pattern would be a surprise nobody asked
			// for.
			v = math.NaN()
		case v == 0:
			// Negative zero is equal to zero, so it groups with zero.
			v = 0
		}
		return binary.LittleEndian.AppendUint64(dst, math.Float64bits(v))
	}
}

// bindBools writes one byte per value.
func bindBools(a *array.Array) writer {
	bits, off := a.Bools(), a.Offset()
	return func(dst []byte, i int) []byte {
		if bits.Get(off + i) {
			return append(dst, 1)
		}
		return append(dst, 0)
	}
}

// bindBytes writes a length and then the bytes.
//
// The length is what stops the two keys "a" and "bc" from encoding the same way
// as "ab" and "c". It is there for the fixed width types too, where it is
// always the same number, because a key made of a string and a decimal needs
// the boundary just as much.
func bindBytes(a *array.Array) writer {
	return func(dst []byte, i int) []byte {
		p := a.Bytes(i)
		dst = binary.LittleEndian.AppendUint64(dst, uint64(len(p)))
		return append(dst, p...)
	}
}
