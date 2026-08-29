package plan

import (
	"math"
	"runtime"
	"sync"
	"time"
	"weak"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Expressions are canonical. Two expressions that say the same thing are the
// same [Expr], whoever wrote them and whenever they were built, because every
// step goes through the table in this file on the way to being made.
//
// That is worth the trouble for one reason. It turns finding a repeated
// subexpression into a pointer comparison. An optimizer that wants to know
// that (price * qty) is written twice in the same select does not have to walk
// two trees and compare them step by step, and does not have to hash them
// either: the two are the same pointer or they are not the same expression.
// The same comparison answers whether a filter has already been pushed down,
// whether a plan is one that has been run before, and whether two expressions
// in a test are equal.
//
// The other half of the bargain is that an expression never changes once it has
// been built. Nothing here writes to one after the constructor returns, which is
// what makes it safe for the table to hand the same expression to two callers at
// once, and what makes it safe to build a plan on one goroutine and run it on
// several others.
//
// Writing a step the table has seen before is a read lock and a map lookup, and
// allocates nothing. That is not free, since hashing a key this size costs
// about what allocating the step did, so a repeated expression is built for
// roughly the same time as before and no memory at all.
//
// Writing a step the table has not seen costs more than a plain allocation
// would, since it is a write lock, an entry, and a cleanup left with the
// runtime to take that entry out again when nobody holds the expression any
// more. That is the right way round. Expressions come from the program rather
// than from the data, so nearly all of them are ones the program has written
// before, and the case that pays is a caller building a genuinely new
// expression per row, which is a microsecond or so a row and gets its memory
// back.

// key is everything that makes one step of an expression different from another
// one. It is the key of the table, so it has to be comparable, and that is the
// only reason it is not simply an [Expr].
//
// The operands are in it as pointers rather than as trees. An operand is
// interned before the step over it can be built, so two steps whose operands are
// equal have those operands as the same pointer, and comparing two keys compares
// the whole tree in one step rather than walking it.
type key struct {
	kind Kind
	name string
	lit  any
	cmp  kernel.CompareOp
	ari  kernel.ArithOp
	dt   string
	l, r *Expr
}

// interned holds the canonical expression for every key that anything still
// refers to. The value is a weak pointer, so an expression built for one query
// and then dropped is collected like any other garbage instead of being kept
// alive by the table that made it. A server that builds an expression per
// request would otherwise grow for as long as it ran.
//
// It is a map behind a lock rather than a [sync.Map], for the ordinary reason
// that a map of the right type is a map of the right type. A sync.Map takes its
// key as an any, and turning a key this size into one costs an allocation on
// every lookup, which is most of what building an expression would then be.
var interned = struct {
	sync.RWMutex
	m map[key]weak.Pointer[Expr]
}{m: make(map[key]weak.Pointer[Expr])}

// intern returns the one expression for a step, building it if this is the
// first time anyone has asked for it.
//
// The step is described rather than built by the caller, so that writing an
// expression the program has already written somewhere else allocates nothing
// at all. The two arguments after the key are the parts of a step the key
// cannot hold as they are: the type a cast is to, which is in the key by name,
// and the value of a literal, which is in the key in whatever form a map key
// can be.
func intern(k key, dt dtype.DataType, lit any) *Expr {
	interned.RLock()
	p, found := interned.m[k]
	interned.RUnlock()
	if found {
		if got := p.Value(); got != nil {
			return got
		}
	}

	interned.Lock()
	defer interned.Unlock()

	// Asked again under the write lock, because between the two locks another
	// goroutine may have put the same step in, and the whole point is that
	// there is one of it.
	if have, ok := interned.m[k]; ok {
		if got := have.Value(); got != nil {
			return got
		}
	}

	e := build(k, dt, lit)
	p = weak.Make(e)
	interned.m[k] = p

	// The cleanup is handed the key and the weak pointer rather than closing
	// over the expression, because a cleanup that refers to the thing it is
	// cleaning up keeps it alive and so never runs.
	runtime.AddCleanup(e, forget, deadEntry{k, p})
	return e
}

// build makes the expression a key describes. It is also what a step that does
// not belong in the table is built with, so that the expression is the same
// either way.
func build(k key, dt dtype.DataType, lit any) *Expr {
	return &Expr{
		kind: k.kind,
		name: k.name,
		lit:  lit,
		cmp:  k.cmp,
		ari:  k.ari,
		dt:   dt,
		l:    k.l,
		r:    k.r,
	}
}

// deadEntry is what the cleanup of a collected expression needs to find its
// entry again, which is the key it went in under and which expression was under
// it.
type deadEntry struct {
	k key
	p weak.Pointer[Expr]
}

// forget takes a collected expression out of the table.
//
// The entry is only removed when it is still the one this expression put there.
// An expression can be collected after another one has already replaced its
// entry, which happens when the second was built while the first was on its way
// out, and removing the entry then would lose an expression that is still in
// use.
func forget(e deadEntry) {
	interned.Lock()
	defer interned.Unlock()

	if p, ok := interned.m[e.k]; ok && p == e.p {
		delete(interned.m, e.k)
	}
}

// bytesKey is a binary literal in the form a map key can be. It is a named type
// and not a plain string so that the bytes "AAPL" and the string "AAPL" are two
// literals, which they are: one is a binary column and the other is a utf8 one.
type bytesKey string

// litKey returns the form of a literal that can be a map key, and false for the
// literals that have no such form.
//
// The types listed here are the ones a column can hold. Anything else is a value
// that will be turned away when the expression is bound to a frame, and it is
// not put in the table in the meantime, because hashing a value of a type that
// does not support equality panics and the caller's mistake should be an error
// rather than that.
func litKey(v any) (any, bool) {
	switch v := v.(type) {
	case nil, bool, string, time.Time:
		return v, true
	case int, int8, int16, int32, int64:
		return v, true
	case uint, uint8, uint16, uint32, uint64:
		return v, true
	case float32:
		// A NaN is not equal to itself, so a key holding one can never be
		// found again and never be deleted again either. Two NaN literals are
		// two expressions instead, which is the only place in the package where
		// an expression is not shared with an equal one.
		return v, !math.IsNaN(float64(v))
	case float64:
		return v, !math.IsNaN(v)
	case []byte:
		return bytesKey(v), true
	default:
		return nil, false
	}
}
