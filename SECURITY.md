# Security and data policy

This repository contains source code, small JSON/CSV summaries, and generated notebooks only.

Never commit Kaggle API files, browser/session data, passwords, tokens, competition data, participant submissions, model checkpoints, generated frontiers, or third-party tablebases. The repository validator rejects common credential names, secret-like text, model weights, archives, and oversized files. Kaggle notebooks discover competition data and model assets at runtime under `/kaggle/input` and write outputs only under `/kaggle/working`.

Report accidental exposure privately to the repository owner and rotate the affected credential before removing it from Git history.

