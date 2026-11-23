"""Train + evaluate the one-shot classifier, write the ablation table and the trained model.

Example (Colab, T4):
    python run.py --data /content/data --seeds 0 1 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from oneshot import methods as M
from oneshot.data import build_splits, save_split
from oneshot.encode import Backbone, cached, safe
from oneshot.pipeline import DEFAULTS, backbone_logits, fit_backbone, predict
from oneshot.text import informative, load_classnames

DEFAULT_BACKBONES = ["ViT-SO400M-16-SigLIP2-384:webli", "ViT-H-14-quickgelu:dfn5b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (see README for layouts)")
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES, help="open_clip 'model:pretrained' specs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="which random shot is drawn per class")
    ap.add_argument("--n_aug", type=int, default=64, help="augmented views of each shot")
    ap.add_argument("--classnames", default=None, help="optional file with human-readable class names")
    ap.add_argument("--no_text", action="store_true", help="disable the text branch")
    ap.add_argument("--no_balance", action="store_true", help="transductive step without uniform-class prior")
    ap.add_argument("--no_tta", action="store_true", help="no flip test-time augmentation")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    hp = {k: getattr(a, k) for k in DEFAULTS}
    os.makedirs(a.out, exist_ok=True)

    info = build_splits(a.data, a.seeds)
    classes = info["classes"]
    names = load_classnames(classes, a.classnames)
    use_text = not a.no_text and (a.classnames is not None or informative(classes))
    save_split(info, os.path.join(a.out, "one_shot_split.json"))
    test_paths = [p for p, _ in info["test_items"]]
    y = torch.tensor([classes.index(c) for _, c in info["test_items"]])
    print(f"{len(classes)} classes | train dir {info['train_dir']} ({info['n_train']} imgs) | "
          f"eval on {info['eval_name']}: {len(test_paths)} imgs | text branch: {use_text}")
    print("class names:", names)

    # ---------------- encode everything with frozen backbones ----------------
    X, WT, S = {}, {}, {}
    tag = f"{info['eval_name']}_{len(test_paths)}_{'tta' if not a.no_tta else 'notta'}"
    for spec in a.backbones:
        t0 = time.time()
        bb = Backbone(spec, a.device)
        d = os.path.join(a.cache_dir, safe(spec))
        X[spec] = cached(f"{d}/test_{tag}.pt",
                         lambda: bb.encode_test(test_paths, a.bs, a.workers, tta=not a.no_tta))
        WT[spec] = bb.encode_text(names) if use_text else None
        for s in a.seeds:
            paths = [info["supports"][s][c][0] for c in classes]
            S[spec, s] = cached(f"{d}/support_seed{s}_aug{a.n_aug}.pt",
                                lambda: bb.encode_support(paths, a.n_aug, seed=s))
        bb.free()
        print(f"[{spec}] encoded in {time.time() - t0:.0f}s")

    # ---------------- evaluate every component (ablation) ----------------
    rows = {}

    def log(name, s, L):
        rows.setdefault(name, {})[s] = M.acc(L, y)

    for s in a.seeds:
        models, Xs = [], []
        for spec in a.backbones:
            Xb, Sb, Wt = X[spec], S[spec, s], WT[spec]
            short = spec.split(":")[0]
            if Wt is not None:
                log(f"{short} | zero-shot text", s, M.zero_shot(Xb, Wt))
            log(f"{short} | 1-shot prototype (raw image)", s, M.prototype(Xb, Sb, clean_only=True))
            log(f"{short} | 1-shot prototype (augmented)", s, M.prototype(Xb, Sb))
            log(f"{short} | view cache", s, M.view_cache(Xb, Sb, hp["beta"]))
            m = fit_backbone(Sb, Wt, hp, seed=s)
            log(f"{short} | anchored head", s, M.head_logits(Xb, m["W"]))
            log(f"{short} | head + cache", s, backbone_logits(Xb, m, hp))
            models.append(m)
            Xs.append(Xb)
        log("ENSEMBLE | inductive", s, predict(Xs, models, hp))
        log("ENSEMBLE | + transductive (softmax)", s, predict(Xs, models, hp, True, balanced=False))
        if not a.no_balance:
            log("ENSEMBLE | + transductive (balanced)", s, predict(Xs, models, hp, True, balanced=True))
        if s == a.seeds[0]:
            torch.save(dict(classes=classes, classnames=names, hp=hp, use_text=use_text,
                            backbones=a.backbones, seed=s,
                            support=[info["supports"][s][c][0] for c in classes],
                            models=[{k: v for k, v in m.items()} for m in models]),
                       os.path.join(a.out, "oneshot_model.pt"))

    # ---------------- report ----------------
    lines = ["| Method | " + " | ".join(f"seed {s}" for s in a.seeds) + " | mean ± std |",
             "|---|" + "---|" * (len(a.seeds) + 1)]
    for name, r in rows.items():
        v = np.array([r[s] for s in a.seeds])
        lines.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in v) + f" | **{v.mean():.2f} ± {v.std():.2f}** |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(f"Eval set: {info['eval_name']} ({len(test_paths)} images), {len(classes)} classes, "
                f"1 shot/class, backbones: {', '.join(a.backbones)}\n\nHyper-parameters: {hp}\n\n{table}\n")
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(dict(hp=hp, backbones=a.backbones, seeds=a.seeds, results=rows), f, indent=2)
    print(f"\nSaved model + results to {a.out}/")


if __name__ == "__main__":
    main()
