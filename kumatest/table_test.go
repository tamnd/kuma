package kumatest

import (
	"strings"
	"testing"
)

func TestTable(t *testing.T) {
	tbl := newTable("row", "column", "got", "want")
	tbl.add("1", "price", "150.25", "150.5")
	tbl.add("100", "qty", "4", "5")

	const want = `  row | column | got    | want
------+--------+--------+------
    1 | price  | 150.25 | 150.5
  100 | qty    | 4      | 5`

	if got := tbl.String(); got != want {
		t.Errorf("the table is\n%s\nand it should be\n%s", got, want)
	}
}

// TestTableOfNothing is the shape a table has before anything is added to it.
// Nothing prints one, since a report with no cells leaves the table out, but a
// header on its own should still be a header.
func TestTableOfNothing(t *testing.T) {
	const want = `  row | got | want
------+-----+-----`

	if got := newTable("row", "got", "want").String(); got != want {
		t.Errorf("the empty table is\n%s\nand it should be\n%s", got, want)
	}
}

// TestTableCountsCharactersRatherThanBytes checks the width of a cell holding
// something outside ASCII, which is a value from a real file more often than
// not. The name here is caf\u00e9, which is five bytes and four characters.
func TestTableCountsCharactersRatherThanBytes(t *testing.T) {
	tbl := newTable("row", "got", "want")
	tbl.add("0", "caf\u00e9", "cafe")

	want := "  row | got  | want\n" +
		"------+------+-----\n" +
		"    0 | caf\u00e9 | cafe"

	if got := tbl.String(); got != want {
		t.Errorf("the table is\n%s\nand it should be\n%s", got, want)
	}
}

// TestTableLeavesNoTrailingSpace is worth a test of its own. Trailing
// whitespace in a test log is invisible until something else strips it and two
// people spend an afternoon on why their outputs differ.
func TestTableLeavesNoTrailingSpace(t *testing.T) {
	tbl := newTable("row", "column", "got", "want")
	tbl.add("0", "a", "b", "c")
	tbl.add("1", "aa", "bb", "cc")

	for i, line := range strings.Split(tbl.String(), "\n") {
		if len(line) == 0 {
			t.Errorf("line %d is empty", i)
			continue
		}
		if line[len(line)-1] == ' ' {
			t.Errorf("line %d ends in a space: %q", i, line)
		}
	}
}
