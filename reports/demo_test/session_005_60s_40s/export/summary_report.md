# Observable Learning Participation — Summary

> This report describes observable pose/motion statistics only. It does not
> measure emotion, motivation, intention or internal attention.

- Experiment: `20261126-110817-tcn-adjudicated-v03-600`
- Metric scope: `human_adjudicated_partial`
- Windows: 172 total, 125 accepted, 47 uncertain
- Coverage: 0.7267
- Sessions: session_005

## Class distribution

| Label | Count |
|---|---:|
| HIGH | 57 |
| MODERATE | 63 |
| LOW | 5 |
| UNCERTAIN | 47 |

## Timeline

| Session | Minute | Windows | HIGH | MODERATE | LOW | UNCERTAIN |
|---|---:|---:|---:|---:|---:|---:|
| session_005 | 1 | 172 | 57 | 63 | 5 | 47 |

## Zone summary

| Zone | N | HIGH | MODERATE | LOW | UNCERTAIN |
|---|---:|---:|---:|---:|---:|
| BACK-CENTER | 25 | 7 | 3 | 3 | 12 |
| BACK-RIGHT | 31 | 9 | 6 | 0 | 16 |
| FRONT-RIGHT | 17 | 17 | 0 | 0 | 0 |
| MIDDLE-CENTER | 20 | 0 | 19 | 1 | 0 |
| MIDDLE-LEFT | 47 | 15 | 25 | 0 | 7 |
| MIDDLE-RIGHT | 32 | 9 | 10 | 1 | 12 |

## Limitations

- Observable participation only; no emotion/intention inference.
- 2D pose features; head_orientation_proxy is not gaze.
- No identity, no face recognition, anonymous track ids only.
- Metrics compare model output against the labels recorded in the input (human_adjudicated_partial). Weak-rule metrics are NOT human ground-truth accuracy.
