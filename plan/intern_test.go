package plan

// The tests for the expression table are in the package rather than beside it,
// because what they check is that two expressions are the same pointer, and the
// table itself is not something the package hands out.

import (
	"math"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// priceTimesQty is the expression these tests build over and over, since the
// whole point is that building it twice gives the same thing back.
func priceTimesQty() *Expr {
	return Arith(kernel.OpMul, Col("price"), Col("qty"))
}

func TestEqualExpressionsAreOneExpression(t *testing.T) {
	a := Compare(kernel.OpGt, priceTimesQty(), Lit(int64(100)))
	b := Compare(kernel.OpGt, priceTimesQty(), Lit(int64(100)))

	if a != b {
		t.Fatalf("%s was built twice and came back as two expressions, %p and %p", a, a, b)
	}
}

func TestSubexpressionsAreShared(t *testing.T) {
	// Two different expressions over the same product, which is the case
	// common subexpression elimination exists for.
	a := Compare(kernel.OpGt, priceTimesQty(), Lit(int64(100)))
	b := Compare(kernel.OpLt, priceTimesQty(), Lit(int64(5)))

	if a.Left() != b.Left() {
		t.Errorf("the %s in %s and in %s are two expressions, %p and %p", a.Left(), a, b, a.Left(), b.Left())
	}
	if a.Left().Left() != Col("price") {
		t.Errorf("the price handle in %s is not the one a caller would build", a)
	}
}

// TestDifferentExpressionsAreNotShared is the other half of the promise.
// Sharing an expression with one that is not equal to it would be a wrong
// answer rather than a slow one, so every way two steps can differ is listed.
func TestDifferentExpressionsAreNotShared(t *testing.T) {
	price := Col("price")
	qty := Col("qty")

	cases := []struct {
		what string
		a, b *Expr
	}{
		{"the column name", price, qty},
		{"the operator", Arith(kernel.OpAdd, price, qty), Arith(kernel.OpMul, price, qty)},
		{"the comparison", Compare(kernel.OpLt, price, qty), Compare(kernel.OpLe, price, qty)},
		{"the side an operand is on", Arith(kernel.OpSub, price, qty), Arith(kernel.OpSub, qty, price)},
		{"the kind of step", And(price, qty), Or(price, qty)},
		{"the one sided step", IsNull(price), IsNotNull(price)},
		{"the type cast to", Cast(dtype.Float64, price), Cast(dtype.Float32, price)},
		{"the value of a literal", Lit(int64(1)), Lit(int64(2))},
		{"the type of a literal", Lit(int64(1)), Lit(1.0)},
		{"a literal against nothing", Lit(nil), Lit(int64(0))},
		{"bytes against a string", Lit([]byte("AAPL")), Lit("AAPL")},
	}

	for _, c := range cases {
		if c.a == c.b {
			t.Errorf("%s and %s differ in %s and are the same expression", c.a, c.b, c.what)
		}
	}
}

func TestByteLiteralsAreCopiedAndShared(t *testing.T) {
	b := []byte("AAPL")
	a := Lit(b)
	if got := Lit([]byte("AAPL")); got != a {
		t.Errorf("two literals holding the same bytes are two expressions, %p and %p", a, got)
	}

	// The caller still has the slice. An expression that changed when they
	// wrote to it would be a different expression than the one in the table.
	b[0] = 'B'
	if got, want := a.String(), `"AAPL"`; got != want {
		t.Errorf("after writing to the slice the literal reads %s, want %s", got, want)
	}
}

func TestNaNLiteralsAreNotShared(t *testing.T) {
	nan := func() *Expr { return Lit(math.NaN()) }

	// Not equal to itself, so it cannot be looked up again. Two of them being
	// two expressions is the documented cost of that, and the thing worth
	// testing is that it stays a cost rather than becoming an entry nothing can
	// remove.
	if a, b := nan(), nan(); a == b {
		t.Errorf("two NaN literals came back as one expression at %p", a)
	}

	// The table has to be no bigger for them, since an entry under a key that
	// cannot be found again is an entry that cannot be removed again either.
	everything := func(key) bool { return true }
	before := countKeys(everything)
	held := make([]*Expr, 100)
	for i := range held {
		held[i] = nan()
	}
	if after := countKeys(everything); after != before {
		t.Errorf("building %d NaN literals put %d entries in the table", len(held), after-before)
	}
}

func TestLiteralOfATypeNoColumnHoldsIsNotShared(t *testing.T) {
	// A map cannot be a map key, so hashing one panics. The mistake belongs in
	// an error when the expression is bound to a frame, not in a panic when it
	// is written, so a value like this is kept out of the table.
	type notAValue map[string]int

	a, b := Lit(notAValue{}), Lit(notAValue{})
	if a == b {
		t.Errorf("two literals of an unsupported type came back as one expression at %p", a)
	}
	if _, ok := litKey(notAValue{}); ok {
		t.Errorf("a %T was accepted as a key of the table", notAValue{})
	}
}

func TestAnExpressionNobodyHoldsIsCollected(t *testing.T) {
	const prefix = "collected column "
	const want = 200

	// The expressions are built and dropped inside a function of their own, so
	// that nothing on this stack still refers to them while the collector is
	// being asked about them.
	func() {
		held := make([]*Expr, want)
		for i := range held {
			held[i] = Col(prefix + strconv.Itoa(i))
		}
		if got := countCols(prefix); got != want {
			t.Fatalf("%d of the %d expressions just built are in the table", got, want)
		}
	}()

	// The cleanup that empties an entry runs after a collection rather than
	// during it, so this waits for it instead of assuming one pass is enough.
	deadline := time.Now().Add(30 * time.Second)
	for {
		runtime.GC()
		got := countCols(prefix)
		if got == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d expressions are still in the table, so dropping one does not free it", got, want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestInternIsSafeForConcurrentUse(t *testing.T) {
	const goroutines = 8

	got := make([]*Expr, goroutines)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			got[i] = Compare(kernel.OpGe, priceTimesQty(), Lit(2.5))
		}()
	}
	close(start)
	wg.Wait()

	for i, e := range got {
		if e != got[0] {
			t.Fatalf("goroutine %d built %s at %p and goroutine 0 built it at %p", i, e, e, got[0])
		}
	}
}

// countCols counts the column handles in the table whose name starts with a
// prefix, which is how a test looks at its own expressions and not at the ones
// every other test in the package has left behind.
func countCols(prefix string) int {
	return countKeys(func(k key) bool {
		return k.kind == KindColumn && strings.HasPrefix(k.name, prefix)
	})
}

func countKeys(match func(key) bool) int {
	interned.RLock()
	defer interned.RUnlock()

	n := 0
	for k := range interned.m {
		if match(k) {
			n++
		}
	}
	return n
}

// BenchmarkExprShared is the cost of writing an expression the program has
// already written somewhere else, which is what the table is for and what most
// building does.
func BenchmarkExprShared(b *testing.B) {
	for b.Loop() {
		sink = Compare(kernel.OpGt, priceTimesQty(), Lit(int64(100)))
	}
}

// BenchmarkExprNew is the cost of writing one the program has not, since the
// literal is different every time. This is the price the table charges, and the
// difference between the two is what it pays back.
func BenchmarkExprNew(b *testing.B) {
	i := int64(0)
	for b.Loop() {
		i++
		sink = Compare(kernel.OpGt, priceTimesQty(), Lit(i))
	}
}

// sink keeps the benchmarks above from being optimized away, and keeps the
// expression they built alive until the next one replaces it.
var sink *Expr
