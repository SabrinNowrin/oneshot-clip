# One-Shot Image Classification with Foundation Models (ITCS 6169/8169 – Assignment 3)

16 classes, **exactly one labelled training image per class**. No backbone is fine-tuned; we adapt
frozen SigLIP 2 and DFN-CLIP embeddings with a light, anchored head and test-time adaptation.

## Result

**Best test accuracy: 94.50 %** (400 test images, 16 classes, 1 shot per class; SigLIP 2 + DFN-CLIP
ensemble with transductive test-time adaptation, shot draw seed 0, the model saved in `artifacts/`).

| Setting | seed 0 | mean ± std over 3 shot draws |
|---|---|---|
| Zero-shot text only (SigLIP 2) | 90.00 | 90.00 ± 0.00 |
| Anchored head (SigLIP 2) | 91.50 | 91.83 ± 0.31 |
| Ensemble, inductive | 91.00 | 90.08 ± 1.66 |
| Ensemble + transductive TTA | **94.50** | 91.42 ± 3.36 |

Reproduce the headline number with the saved model:

```bash
python infer.py --model artifacts/oneshot_model.pt --images data/test --transductive --no_balance
# -> accuracy: 94.50%
```

Full ablation: [`artifacts/results.md`](artifacts/results.md). The exact images used as the single
shots are listed in [`artifacts/one_shot_split.json`](artifacts/one_shot_split.json).

## Method in one picture

```
 1 image / class ──► 64 random augmentations ──► frozen image encoder ──► support views S [16 × 65 × D]
 class names ──► 16 prompt templates ─────────► frozen text encoder  ──► text weights Wt [16 × D]

 per backbone:  W0 = norm(Wt + mean(S))      (text prior + visual prototype)
                W  = argmin CE(S; W) + λ‖W − W0‖²   (mixup, label smoothing, 300 steps, ~1 s)
                logits = cos(x, W) + 0.5 · soft-nearest-view(x, S)   (Tip-Adapter-style cache)
 ensemble:      mean log-softmax over SigLIP2-SO400M and DFN ViT-H/14
 test-time:     10 rounds of Sinkhorn-balanced soft k-means on the unlabelled test embeddings,
                centroids anchored to the shot and the inductive prediction kept as a prior
```

## Reproduce (Google Colab, T4 GPU, ~15 min)

Open `colab.ipynb`, or:

```bash
git clone <THIS_REPO> && cd oneshot-clip
pip install -r requirements.txt
# dataset: open the course Google Drive link, copy the `data` folder (train/ test/) here
python run.py --data data --seeds 0 1 2              # encodes, trains, prints the ablation table
python infer.py --model artifacts/oneshot_model.pt --images data/<test-folder> --transductive
```

`run.py` finds the folders automatically. Accepted layouts are `ROOT/train/<class>/…` with
`ROOT/test/<class>/…` (or `val/`), or a single `ROOT/<class>/…`. With a single folder, evaluation
uses every image except the shots. The class names come from the folder names. If the folders are
named `0…15` or with WordNet ids, pass `--classnames names.txt` (one name per line, in folder
order). Without informative names, the text branch is switched off automatically.

Useful flags: `--backbones` (any `open_clip` `model:pretrained`), `--n_aug`, `--no_text`,
`--no_balance`, `--no_tta`, and every hyper-parameter in `oneshot/pipeline.py:DEFAULTS`.

## Rules compliance

* `oneshot/data.py:select_shots` draws one image per class uniformly at random with a fixed seed.
  No other training image is read, its pixels or its label.
* Hyper-parameters are fixed a priori (see `DEFAULTS`). They are **not** tuned on the test set.
* The transductive step uses only the **unlabelled** test images. The inductive number is reported
  as well.

## Trained model

`artifacts/oneshot_model.pt` (a few MB) holds the trained heads, the support-view cache, the text
weights and the hyper-parameters. The frozen backbones are downloaded from Hugging Face by
`open_clip` at inference time.

## Files

| File | Purpose |
|---|---|
| `run.py` | encode → train → evaluate all ablations over seeds → save model |
| `infer.py` | predictions / accuracy for any folder with the saved model |
| `oneshot/data.py` | dataset discovery, one-shot split |
| `oneshot/encode.py` | frozen encoders, augmentation, TTA, prompt ensembles |
| `oneshot/methods.py` | zero-shot, prototypes, view cache, anchored head, Sinkhorn TTA |
| `oneshot/pipeline.py` | default hyper-parameters, glue |
