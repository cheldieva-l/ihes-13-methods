# Exact full-state Korf-style pilot

This is a direct original-state experiment, not `--shortenseqs`. All18 signed
quarter-layer moves and all54 distinct labels (including centers) are retained.
It is separate from the successful bounded path-fragment shortening pipeline.

Set `ORDINARY333_PUBLIC` to the downloaded competition folder containing
`test.csv` and `puzzle_info.json`, and `TWSEARCH_EXE` to your locally compiled
official Rokicki twsearch executable. The existing `../assets/geometry.json`
is used by default; `ORDINARY333_GEOMETRY` can override it.

Run `python prepare_and_check.py` for adapter/export/shallow checks, then
`python run_pilot.py --input your_baseline.csv --seconds 30 --total-seconds 200`.
The baseline supplies only a depth ceiling and valid fallback. Its solution word
is never passed into the full-state search. Timeout is retryable, not invalidity.

Measured14September2026:1960 adapter checks, all1003 source roundtrips,22 official
move-convention checks and11 shallow direct solutions passed. The six-ID pilot
used2threads/512MiB with depth6 solved-state hashed pruning: onlyID9 solved at
length7; IDs20,49,50,950,999 timed out after30seconds each. Zero new shortenings.
The fallback-inclusive21552 is NOT a complete standalone score from this solver.

Next: stronger factored corner/edge/center pattern databases. The full-corner
table has88,179,840 byte distances. Its distance can be added to the24-center
projection distance because outer turns affect corners, middle turns affect
centers. Separate6-edge pattern bounds must be combined by max unless a valid
cost partition is used. No197GB replication or global optimality claim.
