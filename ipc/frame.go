package ipc

import (
	"encoding/binary"
	"fmt"
	"math"
)

// The encapsulated message format, which is how one Arrow IPC message is
// written down so that the next one can be found after it.
//
// A message is four bytes of 0xFFFFFFFF, then the length of the metadata as a
// little endian int32, then the metadata padded out to a multiple of eight,
// then the body. The continuation bytes are there because the first version of
// this format started with the length, and a reader of the old format sees
// 0xFFFFFFFF as a length of minus one and stops rather than reading garbage.
//
// The padding is what lets a reader use the buffers where they lie. Every
// buffer inside a body is aligned to eight bytes from the start of the body,
// so the body itself has to start on eight as well, and that is only true if
// the metadata in front of it is padded.
//
// A length of zero after the continuation is the end of a stream. It is a
// message that is not there rather than an empty one.

const (
	// The four bytes in front of every message.
	fbContinuation = 0xFFFFFFFF

	// What a message is padded to, and what every buffer inside a body is
	// aligned to.
	fbPad = 8

	// The continuation and the length together, which is what has to be read
	// before a reader knows how much more there is.
	fbPrefix = 8
)

// frame writes the continuation, the length and the message, padded.
func frame(msg []byte) []byte {
	n := len(msg)
	padded := (n + fbPad - 1) &^ (fbPad - 1)

	out := make([]byte, fbPrefix+padded)
	binary.LittleEndian.PutUint32(out, fbContinuation)
	binary.LittleEndian.PutUint32(out[4:], uint32(padded))
	copy(out[fbPrefix:], msg)
	return out
}

// unframe reads one encapsulated message and returns it along with whatever
// follows it, which is the body of the message and then the next one.
//
// A message with nothing after the prefix is the end of a stream, and comes
// back as a nil message and a nil error, so that a caller can tell it apart
// from a message it could not read.
func unframe(b []byte) (msg, rest []byte, err error) {
	if len(b) < fbPrefix {
		return nil, nil, fmt.Errorf("ipc: %w: %d bytes, too few for a message prefix",
			ErrMessage, len(b))
	}

	// The old format with no continuation is still readable: the first four
	// bytes are the length instead, and no length is 0xFFFFFFFF because a
	// message that large would not fit in an int32 to begin with.
	at := 0
	if binary.LittleEndian.Uint32(b) == fbContinuation {
		at = 4
	}
	n := int32(binary.LittleEndian.Uint32(b[at:]))
	at += 4
	if n < 0 {
		return nil, nil, fmt.Errorf("ipc: %w: a message of %d bytes", ErrMessage, n)
	}
	if n == 0 {
		return nil, b[at:], nil
	}
	if int64(at)+int64(n) > int64(len(b)) {
		return nil, nil, fmt.Errorf("ipc: %w: a message of %d bytes with %d left to read it from",
			ErrMessage, n, len(b)-at)
	}
	return b[at : at+int(n)], b[at+int(n):], nil
}

// checkLength refuses a length that does not fit in the int32 the format
// writes it as. A schema of two million columns is somebody generating column
// names in a loop, and saying so is better than writing a length that reads
// back as a negative number.
func checkLength(n int, what string) error {
	if n > math.MaxInt32 {
		return fmt.Errorf("ipc: %w: %s is %d bytes, which does not fit in the int32 length",
			ErrMessage, what, n)
	}
	return nil
}
