# Model Artifacts

Source model definitions live in `src/models/`. This directory is for **trained artifacts/checkpoints references**, not Python model source.

## Checkpoints

Large files such as `*.pt`, `*.pth`, `*.ckpt` are ignored by Git. Recommended storage for Colab training is Google Drive or another approved private storage location.

For each official experiment, commit/version the lightweight metadata under `reports/experiments/<experiment_id>/`:
- config snapshot;
- metrics JSON/CSV;
- training curves/confusion matrix;
- checkpoint filename/reference or private storage location (do not commit secrets/share tokens);
- git commit, dataset version, seed.

Do not commit a checkpoint merely to show that training happened.
