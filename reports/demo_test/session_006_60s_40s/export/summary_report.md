# Observable Learning Participation — Summary

> This report describes observable pose/motion statistics only. It does not
> measure emotion, motivation, intention or internal attention.

- Experiment: `20261126-110817-tcn-adjudicated-v03-600`
- Metric scope: `human_adjudicated_partial`
- Windows: 153 total, 109 accepted, 44 uncertain
- Coverage: 0.7124
- Sessions: session_006

## Class distribution

| Label | Count |
|---|---:|
| HIGH | 40 |
| MODERATE | 66 |
| LOW | 3 |
| UNCERTAIN | 44 |

## Timeline

| Session | Minute | Windows | HIGH | MODERATE | LOW | UNCERTAIN |
|---|---:|---:|---:|---:|---:|---:|
| session_006 | 1 | 153 | 40 | 66 | 3 | 44 |

## Zone summary

| Zone | N | HIGH | MODERATE | LOW | UNCERTAIN |
|---|---:|---:|---:|---:|---:|
| BACK-CENTER | 9 | 2 | 0 | 0 | 7 |
| BACK-RIGHT | 22 | 11 | 0 | 0 | 11 |
| FRONT-RIGHT | 4 | 1 | 3 | 0 | 0 |
| MIDDLE-CENTER | 20 | 0 | 20 | 0 | 0 |
| MIDDLE-LEFT | 45 | 16 | 21 | 2 | 6 |
| MIDDLE-RIGHT | 53 | 10 | 22 | 1 | 20 |

## Limitations

- Observable participation only; no emotion/intention inference.
- 2D pose features; head_orientation_proxy is not gaze.
- No identity, no face recognition, anonymous track ids only.
- Metrics compare model output against the labels recorded in the input (human_adjudicated_partial). Weak-rule metrics are NOT human ground-truth accuracy.
