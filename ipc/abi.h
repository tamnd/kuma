// The two structs of the Arrow C data interface, copied from the
// specification. Every library that speaks this interface has the same
// declarations somewhere, which is the point of it: two libraries that have
// never heard of each other agree on nine fields and a function pointer.
//
// The include guard is the one the specification names. It is spelled that way
// on purpose, so that a program that also includes the header shipped with the
// C++ library gets one copy of these structs rather than two conflicting ones.

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#include <stdint.h>

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
	const char* format;
	const char* name;
	const char* metadata;
	int64_t flags;
	int64_t n_children;
	struct ArrowSchema** children;
	struct ArrowSchema* dictionary;

	void (*release)(struct ArrowSchema*);
	void* private_data;
};

struct ArrowArray {
	int64_t length;
	int64_t null_count;
	int64_t offset;
	int64_t n_buffers;
	int64_t n_children;
	const void** buffers;
	struct ArrowArray** children;
	struct ArrowArray* dictionary;

	void (*release)(struct ArrowArray*);
	void* private_data;
};

#endif
