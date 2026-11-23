"""Dataset discovery and the one-shot split.

Supported layouts (class = sub-folder name):
    ROOT/train/<class>/*.jpg  +  ROOT/test/<class>/*.jpg   (val/ also accepted for test)
    ROOT/<class>/*.jpg        (no test split -> we evaluate on every image NOT used as a shot)
"""
import json
import os
import random

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def _is_img(p):
    return os.path.splitext(p)[1].lower() in IMG_EXT


def _class_dirs(d):
    return sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e)) and not e.startswith("."))


def _has_images(d):
    for c in _class_dirs(d):
        cd = os.path.join(d, c)
        if any(_is_img(f) for f in os.listdir(cd)):
            return True
    return False


def scan(d, classes=None):
    """Return list of (path, class_name) for an ImageFolder-style directory."""
    out = []
    for c in (classes or _class_dirs(d)):
        cd = os.path.join(d, c)
        if not os.path.isdir(cd):
            continue
        for root, _, files in os.walk(cd):
            for f in sorted(files):
                if _is_img(f):
                    out.append((os.path.join(root, f), c))
    return out


def discover(root):
    """Locate train / test folders. Returns (train_dir, test_dir_or_None)."""
    root = os.path.abspath(root)
    # descend through single wrapper folders (common with zip downloads)
    while True:
        subs = _class_dirs(root)
        low = {s.lower(): s for s in subs}
        if "train" in low:
            train = os.path.join(root, low["train"])
            test = None
            for name in ("test", "val", "valid", "validation"):
                if name in low:
                    test = os.path.join(root, low[name])
                    break
            return train, test
        if len(subs) == 1 and not _has_images(root):
            root = os.path.join(root, subs[0])
            continue
        return root, None


def select_shots(train_items, classes, seed, shots=1):
    """Pick `shots` images per class uniformly at random (no peeking at content)."""
    rng = random.Random(seed)
    by_cls = {c: sorted(p for p, y in train_items if y == c) for c in classes}
    sel = {}
    for c in classes:
        if len(by_cls[c]) < shots:
            raise ValueError(f"class {c!r} has only {len(by_cls[c])} images")
        sel[c] = rng.sample(by_cls[c], shots)
    return sel


def build_splits(root, seeds, shots=1):
    train_dir, test_dir = discover(root)
    classes = _class_dirs(train_dir)
    train_items = scan(train_dir, classes)
    supports = {s: select_shots(train_items, classes, s, shots) for s in seeds}
    if test_dir is not None:
        test_items = scan(test_dir, classes)
        eval_name = "test"
    else:
        used = {p for sup in supports.values() for ps in sup.values() for p in ps}
        test_items = [(p, y) for p, y in train_items if p not in used]
        eval_name = "train-minus-shots"
    return dict(train_dir=train_dir, test_dir=test_dir, classes=classes,
                supports=supports, test_items=test_items, eval_name=eval_name,
                n_train=len(train_items))


def save_split(info, path):
    with open(path, "w") as f:
        json.dump({"classes": info["classes"],
                   "supports": {str(k): v for k, v in info["supports"].items()},
                   "eval_set": info["eval_name"], "n_eval": len(info["test_items"])}, f, indent=2)
