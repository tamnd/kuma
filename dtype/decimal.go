package dtype

import "strconv"

// The limits on decimal precision, which come from how many digits fit in the
// underlying integer.
const (
	// MaxDecimal128Precision is the largest precision a Decimal128 can hold.
	MaxDecimal128Precision = 38

	// MaxDecimal256Precision is the largest precision a Decimal256 can hold.
	MaxDecimal256Precision = 76
)

// Decimal128 is an exact fixed point number stored as a 128 bit signed integer
// scaled by ten to the power of Scale.
//
// Precision is the total number of significant digits and Scale is how many of
// them are after the decimal point, so money in pounds and pence is
// Decimal128{Precision: 18, Scale: 2} and the value 12.34 is stored as 1234.
//
// This is what a currency column should be. A float64 cannot represent 0.1, so
// summing a million float prices gives an answer that is close to right and
// disagrees with the ledger, and the report gets rewritten in the accounting
// system instead.
//
// Scale may be negative, which multiplies rather than divides: a Scale of -3
// means the stored integer counts thousands. Arrow allows this and readers in
// the wild produce it.
type Decimal128 struct {
	Precision int32
	Scale     int32
}

// Kind returns Decimal128Kind.
func (t Decimal128) Kind() Kind { return Decimal128Kind }

// String returns the canonical name, such as "decimal128(18, 2)".
func (t Decimal128) String() string { return decimalString("decimal128", t.Precision, t.Scale) }

// Bits returns 128.
func (t Decimal128) Bits() int { return 128 }

// Decimal256 is an exact fixed point number stored as a 256 bit signed integer
// scaled by ten to the power of Scale. It is Decimal128 with room for more
// digits, for the cases where 38 is not enough.
type Decimal256 struct {
	Precision int32
	Scale     int32
}

// Kind returns Decimal256Kind.
func (t Decimal256) Kind() Kind { return Decimal256Kind }

// String returns the canonical name, such as "decimal256(50, 8)".
func (t Decimal256) String() string { return decimalString("decimal256", t.Precision, t.Scale) }

// Bits returns 256.
func (t Decimal256) Bits() int { return 256 }

func decimalString(name string, precision, scale int32) string {
	return name + "(" + strconv.Itoa(int(precision)) + ", " + strconv.Itoa(int(scale)) + ")"
}
