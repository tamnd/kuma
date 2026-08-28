package ipc

// MaxBlock is the size the offset layouts split the data buffer at, so that a
// test can lower it and cover the splitting without the memory it would
// otherwise take.
var MaxBlock = &maxBlock
