# Task 2: Reading Comprehension System

This package implements:
1. **BiDAF with traditional GloVe embeddings**
2. **BiDAF + BERT-Base contextual embeddings**
3. **Training / validation / testing with EM and F1**
4. **GPU-aware training with metric logging and plotting**

## Expected project layout

```text
project_root/
├── code/
│   └── task_2/
│       ├── train.py
│       ├── plot_metrics.py
│       ├── dataset.py
│       ├── models.py
│       ├── utils.py
│       ├── requirements.txt
│       └── README.md
└── data/
    ├── squad/
    │   ├── train-v1.1.json
    │   └── dev-v1.1.json
    └── glove/
        └── glove.6B.100d.txt   # optional, only needed for the GloVe baseline
```

## What you need to download

### Required
Download **SQuAD v1.1** and place the files here:

```text
./data/squad/train-v1.1.json
./data/squad/dev-v1.1.json
```

### Optional but recommended for the comparison baseline
Download **GloVe 6B 100d** and place this file here:

```text
./data/glove/glove.6B.100d.txt
```

If you do not download GloVe, you can still run the **BERT + BiDAF** model.

## Install

```bash
pip install -r requirements.txt
```

## Run from project root

### 1) Train GloVe + BiDAF
```bash
python ./code/task_2/train.py --embedding_type glove
```

### 2) Train BERT + BiDAF
```bash
python ./code/task_2/train.py --embedding_type bert --freeze_bert true
```

### 3) Plot saved metrics
```bash
python ./code/task_2/plot_metrics.py --metrics_file ./code/task_2/runs/<run_name>/metrics_log.txt
```

## Recommended laptop-friendly settings

### GloVe baseline
```bash
python ./code/task_2/train.py \
  --embedding_type glove \
  --batch_size 16 \
  --epochs 8 \
  --max_context_len 300 \
  --max_question_len 30
```

### BERT + BiDAF on 8GB GPU
```bash
python ./code/task_2/train.py \
  --embedding_type bert \
  --freeze_bert true \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --epochs 3 \
  --max_context_len 256 \
  --max_question_len 32
```

If GPU memory is still tight, lower `--batch_size` to `2`.

## Outputs
Each training run creates:

```text
./code/task_2/runs/<run_name>/
├── best_model.pt
├── config.json
├── metrics_log.txt
├── summary.json
├── loss_curve.png        # created by plot_metrics.py
└── score_curve.png       # created by plot_metrics.py
```

## Notes on the comparison required by the task
Run both variants on the same dataset split:
- `--embedding_type glove`
- `--embedding_type bert`

Then compare the saved `summary.json` files for:
- validation EM / F1
- test EM / F1
- training speed
- memory usage

This gives you the analysis required for how BERT embeddings affect performance relative to traditional embeddings.
