# Dataset Splits

Split files are versioned artifacts. Do not random-split frames/windows from the same session across train/val/test.

Recommended files when T18 is implemented:

```text
train_sessions.txt
val_sessions.txt
test_sessions.txt
split_metadata.json
```

`split_metadata.json` should record dataset version, grouping key, seed, timestamp and script/config used.
