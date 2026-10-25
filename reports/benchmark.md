# Benchmark (measured)

- Machine: Windows 10 AMD64
- Python: 3.14.7
- PyTorch: 2.14.1+cpu
- Video: `data\raw\class_1.mp4`
- Pose benchmark frames: 60
- End-to-end max frames: 300 (actual sampled: 150)

## Pose / tracking

| Tracker | Frames | Detections | FPS | Tracks | Median track (s) | Peak Python RAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| bytetrack.yaml | 60 | 627 | 4.33 | 14 | 5.9 | 22.0 |
| botsort.yaml | 60 | 627 | 4.06 | 14 | 5.9 | 21.9 |

ID switches/minute and fragmentation/person: **N/A** — no ground-truth track annotations.

## Temporal models

| Experiment | Model | Params | Checkpoint (MB) | ms / 122 windows | ms / window |
|---|---|---:|---:|---:|---:|
| 20261025-094931-lstm-weaklabel-v0-1 | LSTM | 246531 | 2.84 | 26.19 | 0.215 |
| 20261025-101644-tcn-weaklabel-v0-1 | TCN | 63363 | 0.76 | 15.57 | 0.128 |
| 20261025-102826-lstm-posefeat-v2 | LSTM | 255235 | 2.94 | 39.76 | 0.326 |
| 20261025-162855-lstm-full-v0-1 | LSTM | 246531 | 2.84 | 31.74 | 0.26 |
| 20261025-163801-tcn-full-v0-1 | TCN | 63363 | 0.76 | 17.05 | 0.14 |

> These are single-run CPU measurements. They are not a scientific comparison of tracker quality; run with the same hardware and multiple repetitions before quoting.

## End-to-end (pose → features → model → export)

| Video | Sampled frames | Windows | Accepted | Elapsed (s) | E2E FPS | s per video minute |
|---|---:|---:|---:|---:|---:|---:|
| class_1.mp4 | 150 | 66 | 63 | 23.955 | 6.26 | 95.82 |
