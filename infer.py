"""Inference with the saved one-shot model.

    python infer.py --model artifacts/oneshot_model.pt --images /content/data/test [--transductive]

--images may be an ImageFolder-style directory (class sub-folders -> accuracy is reported)
or a flat folder of images (predictions only). Writes predictions.csv.
"""
import argparse
import csv
import os

import torch

from oneshot import methods as M
from oneshot.data import IMG_EXT, discover, scan
from oneshot.encode import Backbone
from oneshot.pipeline import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/oneshot_model.pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--transductive", action="store_true", help="test-time adaptation over all given images")
    ap.add_argument("--no_balance", action="store_true")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.model)
    classes = ck["classes"]
    items = scan(a.images, classes)
    if not items:  # maybe a dataset root: use its test split if it has one
        _, test_dir = discover(a.images)
        if test_dir:
            a.images = test_dir
            items = scan(test_dir, classes)
    if items:
        paths, y = [p for p, _ in items], torch.tensor([classes.index(c) for _, c in items])
    else:
        paths = sorted(os.path.join(a.images, f) for f in os.listdir(a.images)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        y = None
    print(f"{len(paths)} images")

    Xs = []
    for spec in ck["backbones"]:
        bb = Backbone(spec, a.device)
        Xs.append(bb.encode_test(paths, a.bs))
        bb.free()
    L = predict(Xs, ck["models"], ck["hp"], transductive=a.transductive, balanced=not a.no_balance)
    pred = L.argmax(-1)
    if y is not None:
        print(f"accuracy: {M.acc(L, y):.2f}%")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "pred_class", "confidence"])
        conf = L.softmax(-1).max(-1).values
        for p, k, c in zip(paths, pred.tolist(), conf.tolist()):
            w.writerow([p, classes[k], f"{c:.4f}"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
