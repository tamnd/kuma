# Contributing

Thanks for looking. This document is short on purpose. The long version of what the quality bar means is [docs/09-quality-bar.md](docs/09-quality-bar.md).

## Before you start

The project is early enough that the useful contributions are the ones already written down. Look at the [milestones](https://github.com/tamnd/kuma/milestones), pick the one that is currently open, and take something from its checklist. Anything labelled `good first issue` is genuinely one, meaning it is self contained and does not require holding the whole design in your head.

If you want to work on something, say so on the issue first. Not for gatekeeping, but because the milestones have a deliberate order and a pull request that implements M7 while M3 is still being designed will need rewriting.

If you want to change something in `docs/`, open an issue rather than a pull request. Those documents are the design, and changing them is a decision rather than an edit.

## The rules that are not negotiable

**Correctness before speed.** The scalar kernels are the specification that the vectorized ones are checked against. A fast kernel with no scalar twin will not be merged, however good the numbers are.

**No `internal/`.** This is a constraint on the project and it is paid for by unexporting aggressively. The default answer to "should this be exported" is no.

**The library builds and passes with no `GOEXPERIMENT` set.** SIMD is an optimization and never a dependency. If your change makes the plain build fail or skip tests, it is not finished.

**No `simd` or `archsimd` type in an exported signature.** That API is unstable and its amd64 surface already broke once. Keeping it inside `kuma/kernel` is what limits the cost of the next break to an afternoon.

**Every exported identifier has a doc comment starting with its name.** Enforced by revive in CI, with no exceptions list.

**Nothing panics across an API boundary.** Errors are values. Panics are for programmer error in unexported code, and even there returning an error is usually better.

**No new dependency without a written justification in the pull request.** The core is standard library only. That is what makes `kuma/bitmap` and `kuma/kernel` credible as packages someone might import on their own.

## Tests

Write the test first for anything that packs bits, parses bytes or vectorizes a loop. The failure mode in all three cases is silent and lives in the tail, not the middle of the loop, and a fuzzer finds it in seconds while a hand written table test finds it in production.

What CI runs on every pull request:

```
go build ./...
go test -race -shuffle=on ./...          on linux, macos and windows, amd64 and arm64
go test ./... with GOEXPERIMENT=simd     on linux and macos
go vet ./... and gofmt -l
golangci-lint run
govulncheck ./...
30 seconds of fuzzing per target
a cross compile across seven targets
the pyarrow cross check
```

Nightly adds twenty minutes of fuzzing per target, the microbenchmarks, and the race suite twenty times over.

Reproduce all of that locally with:

```
go test -race -shuffle=on ./...
go vet ./...
gofmt -l .
golangci-lint run
```

If a fuzzer finds something, commit the failing input in `testdata/fuzz/`. That is what turns a one time discovery into a permanent regression test.

`TestPyarrow` in the `ipc` package builds a shared library and lets pyarrow hand columns to kuma and take them back, which is the only way to check an ABI. It skips unless `python3` can import pyarrow, so install it with `pip install pyarrow` if you are touching that code. CI always runs it.

## Style

Standard Go, `gofmt` formatted, and read the surrounding code before adding to it.

Prefer a second function over a boolean parameter. `SortBy` and `SortDesc`, not `Sort(desc bool)`. Where an operation genuinely has more than three knobs, use an options struct.

Make zero values useful. `var b bitmap.Bitmap` is an empty bitmap, not a panic waiting to happen.

Comments explain why, not what. The code already says what it does.

Error strings are lowercase with no trailing punctuation, and wrapping uses `%w`.

## Commits and pull requests

Commit messages follow the Go convention: `package: short description in the imperative`, a blank line, then the reasoning if it is not obvious. For example:

```
bitmap: clear padding bits after Not

Not inverted the whole final byte including the bits past the length,
which made CountOnes over-report by up to seven for any bitmap whose
length was not a multiple of eight.
```

Keep pull requests to one thing. A change that fixes a bug and also renames four files is two pull requests, and reviewing it as one is worse for both of us.

Link the issue. If there is not one, open it first, because that is where the design discussion goes and a pull request is a bad place for it.

## Getting set up

You need Go 1.27 or newer. Nothing else.

```
git clone https://github.com/tamnd/kuma
cd kuma
go test ./...
```

If `go test` complains about the Go version, that is the toolchain directive doing its job. Either upgrade or let Go download 1.27 for you, which it will do automatically.

## Reporting a bug

Include the Go version, the operating system and architecture, whether `GOEXPERIMENT=simd` was set, and the smallest input that reproduces it. For anything involving a data file, the schema matters more than the data, so a description of the columns and dtypes is usually enough.

## License

By contributing you agree that your contribution is licensed under Apache 2.0, the same as the rest of the project.
