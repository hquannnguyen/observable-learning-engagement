# AGENTS.md — CVF25PRJ10 Project Context & Working Guide

> Tài liệu này dành cho **AI coding agents và thành viên mới**.  
> Mục tiêu: đọc file này trước khi sửa code để hiểu đúng project, tránh refactor/suy luận sai scope và biết cách thực hiện task.

---

## 1. Project identity

- **Group:** CVF25PRJ10
- **Task code:** CVFP26BCPPRJC012
- **Vietnamese title:** Phân tích mức độ tham gia học tập từ tín hiệu hành vi quan sát được
- **English working name:** Observable Learning Participation Analysis
- **Primary stack:** Python, OpenCV, Ultralytics YOLOv8-Pose, tracking, PyTorch LSTM/TCN, Streamlit

### Team

| Name | Student ID | Primary responsibility |
|---|---|---|
| Vũ Văn Phong | B23DCPT250 | Data, annotation, dataset QA |
| Nguyễn Hữu Quân | B23DCPT262 | Pose, tracking, preprocessing, integration |
| Vũ Đức Tài | B23DCPT277 | LSTM/TCN, training, evaluation |
| Nguyễn Dũng Tiến | B23DCPT306 | Streamlit, anonymization, reporting |

---

## 2. What this system is — and is not

The system estimates **observable learning participation** from body pose/motion over time.

It is **not** a system for directly determining:
- emotion,
- boredom,
- motivation,
- intent,
- internal attention,
- comprehension,
- personality,
- identity.

Never rewrite project copy/UI/report to claim that LOW means a student is “lazy”, “bored”, “not paying attention”, etc.

Preferred wording:
- `observable participation`
- `mức độ tham gia học tập quan sát được`
- `pose quality`
- `head-orientation proxy`
- `anonymous track ID`

---

## 3. Canonical pipeline

```text
Video / Camera
    ↓
Video Validation + Frame Sampling
    ↓
YOLOv8-Pose (pretrained pose extractor)
    ↓
ByteTrack (baseline tracker)
    ↓
Pose Quality Gate
    ↓
Keypoint Cleaning + Normalization
    ↓
Pose + Motion Features
    ↓
Sliding Window
    ↓
Customized LSTM / TCN (trained by this project)
    ↓
Canonical 3-class probabilities
    ↓
Uncertainty Gate
    ↓
Canonical labels + UNCERTAIN
    ↓
Display mapping for UI/report
    ↓
Time + Classroom Zone Aggregation
    ↓
Anonymized Streamlit Dashboard + Export
```

**Important:** YOLOv8-Pose is a pretrained extractor. The **customized model** required by the project is the temporal LSTM/TCN classifier trained/evaluated on the project dataset.

---

## 4. Label contract

### Model training target — exactly 3 classes

```text
HIGH_OBSERVED_PARTICIPATION
MODERATE_OBSERVED_PARTICIPATION
LOW_OBSERVED_PARTICIPATION
```

### Quality/annotation state

```text
VALID
OCCLUDED
TRANSITION
INVALID
```

### Canonical persisted/model/API labels

```text
HIGH_OBSERVED_PARTICIPATION
MODERATE_OBSERVED_PARTICIPATION
LOW_OBSERVED_PARTICIPATION
UNCERTAIN   # gate output only, never training target
```

### Display labels only

```text
HIGH / MODERATE / LOW / UNCERTAIN
```

Mapping source of truth: `src/contracts/labels.py`. Never persist short HIGH/MODERATE/LOW values in annotation/window datasets.

Every VALID annotation/window label must carry `label_source` (`HUMAN`, `WEAK_RULE`, `ADJUDICATED`). `WEAK_RULE` additionally requires `label_rule_version`. Weak labels must never be silently mixed into human/adjudicated test ground truth.

`UNCERTAIN` is **not a fourth behavioral class**. It is a system state produced when input quality or model confidence is insufficient.

Do not silently change this contract.

---

## 5. Current repository status

### Completed
- **T01:** scope, input/output, terminology and inference limits.
- **T02:** repository skeleton, baseline config, experiment metadata/logging utilities.

### Current implementation level
This repository is still a **baseline skeleton**. Placeholder directories do not imply their algorithms are implemented.

As of the current context:
- YOLOv8-Pose extraction: not yet considered complete.
- Tracking implementation: not yet considered complete.
- Feature extraction: not yet considered complete.
- LSTM/TCN training: not yet considered complete.
- Full Streamlit flow: not yet considered complete.

Never claim later tasks are complete just because a folder/file placeholder exists.

### Immediate next work
- T03 Data collection/privacy protocol.
- T04 Annotation guide v0.1.
- Then T05 pilot recording setup.

Full task order and DoD: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

---

## 6. Files to read before making changes

Before implementing a task, read in this order:

1. `AGENTS.md`
2. `README.md`
3. `docs/scope.md`
4. `docs/IMPLEMENTATION_PLAN.md`
5. `docs/data_collection_protocol.md` / `docs/annotation_guide.md` / `docs/dataset_schema.md` when data-related
6. `configs/baseline.yaml`
7. `docs/experiment_protocol.md`
8. Task-specific files/code

When documents conflict:
1. Explicit current task/user instruction wins.
2. `docs/scope.md` wins for semantic scope/labels.
3. `docs/IMPLEMENTATION_PLAN.md` wins for task order/DoD.
4. `configs/baseline.yaml` is the baseline configuration, not an immutable research result.

---

## 7. Repository map and responsibilities

```text
configs/
  baseline.yaml              # stable starting configuration
  experiments/               # one config per experiment/change

data/
  raw/                       # raw videos; never commit personal raw video publicly
  annotations/               # human/weak annotations
  poses/                     # pose/tracking outputs
  features/                  # window/features datasets
  splits/                    # fixed train/val/test split definitions

docs/
  scope.md                   # semantic contract
  IMPLEMENTATION_PLAN.md     # task plan and Definition of Done
  experiment_protocol.md     # reproducibility rules

src/
  pose/                      # YOLO pose + trackers
  preprocessing/             # quality, cleaning, normalization, features, windows
  models/                    # LSTM / TCN
  training/                  # datasets/train/eval
  uncertainty/               # quality/model confidence gating
  reporting/                 # aggregation, bias analysis, exports

app/
  streamlit_app.py           # UI entry point

scripts/
  check_setup.py
  new_experiment.py
  ... task-oriented CLIs

reports/experiments/         # experiment artifacts
```

Do not put training logic into the Streamlit file or data-processing logic into UI modules. Keep responsibilities separated.

---

## 8. Engineering rules for AI Agents

### 8.1 Minimal-change rule
The codebase should **not be rewritten or refactored broadly without need**.

For each task:
- change only files necessary for the task,
- preserve working interfaces,
- do not rename/move modules unless required,
- do not replace chosen libraries just because another library is preferred,
- explain any dependency-breaking change.

This rule is especially important for AI-generated code.

### 8.2 No speculative implementation
Do not invent:
- dataset statistics,
- performance metrics,
- model accuracy,
- completed experiments,
- available raw videos,
- annotation agreement,
- hardware results.

If data/results do not exist, mark them `TODO`, `N/A`, or provide code to measure them.

### 8.3 Baseline is a stable reference
`configs/baseline.yaml` represents the project baseline.

Do not repeatedly edit it for every experiment.
Create:

```text
configs/experiments/<experiment>.yaml
```

for variants such as:
- LSTM changes,
- TCN configs,
- window 2/4/6 seconds,
- threshold changes,
- tracker comparison.

Only update `baseline.yaml` when the team intentionally changes the official baseline.

### 8.4 Reproducibility first
Every important training/evaluation run must record:
- experiment ID,
- config snapshot,
- random seed,
- git commit,
- dataset version,
- split version,
- metrics/artifacts.

Use `scripts/new_experiment.py` and follow `docs/experiment_protocol.md`.

### 8.5 Test before claiming done
A task is not Done until its DoD in `docs/IMPLEMENTATION_PLAN.md` is satisfied.

At minimum:
- run syntax/import/smoke test,
- run task-specific verification,
- report errors honestly,
- do not mark a placeholder as implemented.

---

## 9. Data and privacy rules

Raw classroom video can contain personal data.

Mandatory rules:
- Do not commit raw participant video to a public repository.
- Do not create `track_id → name/MSSV` mapping unless explicitly required and approved.
- Prefer pose/features for working datasets after extraction.
- Public/demo output should use anonymization: skeleton/anonymous ID and/or head blur.
- Audio is not needed for baseline; avoid collecting it unless necessary.
- Record consent/permission and retention/access rules in the data protocol.

Never implement face recognition as a “helpful addition”; it is out of scope.

---

## 10. Dataset split rules

Never randomly split frames/windows from the same recording across train and test.

Preferred grouping:
1. participant-level split when participant identity is consistently available for dataset management, or
2. session/day-level split.

Rules:
- Test sessions must be isolated from training.
- Test split is frozen before final model tuning.
- Do not relabel test examples simply because the model performs poorly.
- Any legitimate annotation correction must be logged/versioned.

---

## 11. Pose/tracking contract

Expected per-frame data should support fields equivalent to:

```text
session_id
camera_id
frame_idx
timestamp_ms
track_id
bbox_x1/y1/x2/y2
bbox_conf
17 × (keypoint_x, keypoint_y, keypoint_conf)
visible_ratio
zone_id
```

### Missing keypoints
- Do not encode an unseen point as a valid `(0, 0)` coordinate.
- Keep missing/visibility masks.
- Interpolate only short gaps and retain quality information.

### Pose quality gate
Initial config lives in `baseline.yaml`:
- minimum visible ratio,
- minimum mean keypoint confidence,
- minimum track duration.

Thresholds are initial values, not final scientific results. Tune only through validation experiments.

---

## 12. Feature contract

Candidate feature groups:

- normalized keypoints,
- torso/body angle,
- shoulder slope,
- elbow/arm geometry,
- relative wrist/head/shoulder positions,
- `head_orientation_proxy`,
- velocity/acceleration/motion energy,
- posture stability,
- visible ratio / confidence / missing mask.

Do not call 2D head geometry “gaze” unless a dedicated gaze estimator and ground truth are actually introduced.

Keep feature schema/version deterministic so the same source data generates the same training input.

---

## 13. Temporal model rules

### LSTM and TCN
Both should consume equivalent temporal inputs for fair comparison.

Default sequence config currently starts at:
- sample FPS: see `configs/baseline.yaml`,
- window: 4 seconds,
- stride: 2 seconds.

Ablation may test 2/4/6 seconds. Choose final values from validation results, not test results.

### Imbalanced classes
Prefer:
- class-weighted loss, and/or
- controlled sampler,

while keeping Macro-F1 as a primary metric.

---

## 14. Uncertainty contract

Use two distinct sources:

### Input uncertainty
Examples:
- low visible ratio,
- poor pose confidence,
- too many missing points,
- track too short,
- out-of-view/occluded.

### Model uncertainty
Example:

```python
if max_probability < threshold:
    output = "UNCERTAIN"
```

Threshold must be selected on validation data. Report:
- coverage,
- performance on accepted predictions,
- uncertain rate.

Do not train `UNCERTAIN` as a normal fourth class in the main model unless the project plan is explicitly changed.

---

## 15. Weak supervision rule

Weak supervision is allowed to help reduce annotation effort.

Example labeling functions may use observable signals such as:
- `HAND_RAISED`,
- `READING_WRITING`,
- `PEER_ORIENTED`,
- class phase,
- visibility quality.

But:
- a weak label is not ground truth,
- rule-generated labels must be marked separately,
- human labels and weak labels must be distinguishable in data schema,
- do not evaluate a rule/model against labels generated by the same rule and present it as independent evidence.

---

## 16. Evaluation contract

### Primary classification metrics
- Macro-F1.
- Per-class Precision/Recall/F1.
- Balanced Accuracy.
- Confusion Matrix.

### With uncertainty
- Coverage.
- Accepted prediction performance.
- Uncertain rate.

### Tracking
- ID switches/minute.
- Fragmentation/person.
- Missed detection ratio.
- FPS.

### Positional/viewpoint analysis
For each zone/camera group, report sample count `N` alongside metrics.

Do not claim a location-related performance difference without noting sample size, pose quality and occlusion differences.

---

## 17. Streamlit/output contract

Full-flow UI should eventually include:

1. Upload / config.
2. Processing status.
3. Anonymized preview.
4. Timeline distribution.
5. Classroom zone heatmap.
6. Uncertainty/quality information.
7. Export.

Expected export artifacts may include:

```text
results.csv
timeline.csv
zone_summary.csv
summary.json
summary_report.*
```

A report should describe observable statistics and limitations, not judge individual students.

---

## 18. Performance benchmarking

When full-flow implementation exists, measure and record:
- pose FPS,
- end-to-end FPS,
- seconds per minute of video,
- inference latency,
- peak RAM,
- peak VRAM when applicable,
- model size,
- test hardware/config.

Never invent benchmark values.

---

## 19. Suggested branch/commit workflow

Branch examples:

```text
feature/T06-pose-extractor
feature/T07-bytetrack
feature/T22-lstm
experiment/T28-window-ablation
fix/T14-keypoint-mask
```

Commit examples:

```text
feat(T06): add YOLOv8 pose extraction pipeline
feat(T22): implement configurable LSTM classifier
exp(T28): evaluate 2s 4s 6s windows
fix(T14): keep missing keypoint mask during normalization
docs(T04): refine annotation examples
```

Keep one logical task/change per commit where practical.

---

## 20. How an AI Agent should execute a task

When asked, for example, “complete T06”:

1. Find T06 in `docs/IMPLEMENTATION_PLAN.md`.
2. Read its dependencies and DoD.
3. Inspect existing files; do not assume placeholders contain logic.
4. Implement the smallest coherent change.
5. Add/update tests or smoke checks.
6. Run verification.
7. Report exactly:
   - files changed,
   - command to run,
   - verification result,
   - remaining limitations.
8. Do **not** automatically implement T07/T08 unless needed as a strict dependency or explicitly asked.

If blocked by missing real data, create a testable interface/sample fixture where appropriate and clearly state that real-data validation remains pending.

---

## 21. Definition of a safe/helpful AI change

A good change:
- matches the requested task,
- preserves project semantics,
- is reproducible,
- has a clear run/test command,
- does not claim unmeasured results,
- does not weaken privacy,
- does not add unnecessary complexity.

A bad change:
- rewrites unrelated modules,
- changes labels casually,
- introduces emotion/face recognition,
- random-splits frames into train/test,
- reports fake accuracy,
- edits baseline config for every experiment,
- hides errors instead of surfacing them.

---

## 22. Quick commands

Check repository skeleton:

```bash
python scripts/check_setup.py
```

Create experiment:

```bash
python scripts/new_experiment.py \
  --name lstm-baseline \
  --seed 42 \
  --dataset-version v0.1 \
  --split-version split-v0.1
```

Later, task-specific CLIs should follow the same style and be documented in README/docs.

---

## 23. Source-of-truth checklist

Before answering project questions or modifying code, verify against:

- [ ] `docs/scope.md` — is the proposed behavior in scope?
- [ ] `docs/IMPLEMENTATION_PLAN.md` — which task owns it?
- [ ] `configs/baseline.yaml` — what is baseline vs experiment?
- [ ] `docs/experiment_protocol.md` — is the experiment reproducible?
- [ ] privacy rules — does it expose identity/raw data?

If unsure, preserve the current behavior and propose the minimal necessary change rather than redesigning the project.


---

## 24. Colab / notebook rule

Google Colab is an execution environment, not a separate codebase.

- Notebook files live in `notebooks/`.
- Core pose/preprocessing/model/training/evaluation logic belongs in `src/`.
- A notebook should import/call reusable project code rather than reimplement it.
- Large datasets/checkpoints stay in approved private storage (for example Google Drive); do not commit raw private video or large model files.
- Official train/evaluation runs still require experiment ID, config snapshot, seed, git commit and dataset version.

Current notebook files are templates. Their existence does not mean T06/T22/etc. are implemented.

## 25. Data-contract documents

Before T05+ data work, use these files as contracts:

- `docs/data_collection_protocol.md`
- `docs/annotation_guide.md`
- `docs/dataset_schema.md`
- `docs/privacy_and_consent.md`
- `data/sample/*`

T03/T04 drafts require team review before being marked Done.
