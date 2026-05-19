# Data

Raw videos are not included in this public repository. The original local
experiment uses private short walking clips, so the repository only publishes
static screenshots and aggregate training outputs.

File names should follow the project convention:

```text
<class-id>-<view-or-sequence-id>-<take-id>.mp4
```

For example, `001-01-1.mp4` is assigned to class `001`. The training script reads
the class label from the prefix before the first hyphen.

Place private videos under `data/private/` locally, or pass another folder with
`python scripts/train.py --videos path\to\videos`. Generated tensor caches,
trained weights, and full video data are intentionally excluded from Git.
