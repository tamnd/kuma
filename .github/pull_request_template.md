## What this changes

<!-- One paragraph. If it takes more than that, it is probably two pull requests. -->

Fixes #

## Checklist

- [ ] `go test -race -shuffle=on ./...` passes
- [ ] `go vet ./...` and `gofmt -l .` are clean
- [ ] `golangci-lint run` is clean
- [ ] Builds and passes with no `GOEXPERIMENT` set
- [ ] Every new exported identifier has a doc comment starting with its name
- [ ] New kernels have a scalar twin and a differential fuzz test against it
- [ ] Any fuzz failure found is committed to `testdata/fuzz/`
- [ ] No new dependency, or a justification below
