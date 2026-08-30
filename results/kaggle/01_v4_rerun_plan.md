# Method 1 version 4 fail-closed resume plan

Version 3 completed successfully on Kaggle T4 x2 and produced a replay-valid 1,003-row submission, but it did not advance the aggregate. Its requested Version 2 source was a cancelled run, `resumed_puzzle_ids` was empty, and it executed IDs 100–102 again. This is retained as a verified non-advancing failed shard in `01_v3_nonadvancing.json`.

The corrected protocol is:

1. Do not launch while the higher-priority IHES dual-model task needs Kaggle capacity.
2. Pin successful Method 1 Version 3 as the only kernel source. Cancelled versions are not accepted as resume sources.
3. Before beam search, strictly validate `benchmark_rows.json` and `submission.partial.csv`, including identity, row uniqueness, exact replay, all 1,003 submission rows, and the pinned SHA-256 values `a1255dee599dadd4413b65c6207fa017866e71db825cbc90d361ce5c4a7dd049` and `5a2618f2fbfbc6c951865d96425bc7c5f9c3db71298aee73ad112bcdc3df79b6`.
4. Fail before search unless the completed replay-valid resume IDs are exactly `100, 101, 102`. A zero-row resume is always an error.
5. Retain beam width 365000, model `1778521793`, checkpoint `b19eb25b...`, and the exact two-frame direct/reverse transform portfolio. Execute exactly the next three IDs `103, 104, 105`.
6. Emit combined rows, the resume audit, a replay-valid 1,003-row `submission.csv`, and a pending aggregate verdict. Later shards pin the immediately previous successful advancing version and add at most three missing IDs.

Version 4 is launched only after local tests, repository validation, a public GitHub push, an inactive priority notebook, and a capacity check.
