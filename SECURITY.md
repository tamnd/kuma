# Security

## Supported versions

Nothing is supported yet. The library is pre 1.0 and pre useful. Once there is
a 1.0, the current minor release will get security fixes.

## Reporting

Use GitHub's private vulnerability reporting on the Security tab. Do not open a
public issue for anything exploitable.

The things most likely to be worth reporting once there is code to attack: the
parsers, meaning CSV, NDJSON, Parquet and Arrow IPC, since they read untrusted
bytes and do bounds arithmetic; and the C ABI in `kuma/capi`, since it takes
pointers from a foreign runtime.

Expect a first response within a week.
