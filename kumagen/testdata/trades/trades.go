// Package trades is the package kumagen is run over in the tests. It is under
// testdata so that the go tool leaves it alone, and TestGeneratedCodeCompiles
// copies it into a module of its own to check that what comes out of it builds.
package trades

import (
	"time"

	clock "time"
)

//go:generate kumagen -type Trade

// Trade has a field of every type there is a handle for, tagged and untagged,
// and two fields that are left out.
type Trade struct {
	Symbol  string    `kuma:"symbol"`
	Price   float64   `kuma:"price"`
	Qty     int64     `kuma:"qty"`
	Filled  bool      `kuma:"filled"`
	TS      time.Time `kuma:"ts"`
	OrderID int64
	Venue   string `kuma:"-"`
	Seen    clock.Time
	note    string
}

// Bar is a second type in the same package, so that a run naming both writes
// two files.
type Bar struct {
	Symbol string  `kuma:"symbol"`
	Volume float64 `kuma:"volume"`
	N      int64   `kuma:"n"`
}

// Quote embeds the time it was taken at, so that the name of an embedded field
// is the name of its type.
type Quote struct {
	time.Time
	Bid float64 `kuma:"bid"`
	Ask float64 `kuma:"ask"`
}
