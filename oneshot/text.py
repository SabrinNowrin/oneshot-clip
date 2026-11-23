"""Class-name cleaning and prompt ensembles for the text branch."""
import re

# Generic prompt ensemble (subset of the CLIP ImageNet ensemble that transfers well).
TEMPLATES = [
    "a photo of a {}.",
    "a good photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a low resolution photo of a {}.",
    "a blurry photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a rendering of a {}.",
    "a drawing of a {}.",
    "art of the {}.",
    "itap of a {}.",
    "a {} in the wild.",
]


def clean_name(name):
    n = re.sub(r"^\d+[\s._-]+", "", name)          # "03_golden_retriever" -> "golden_retriever"
    n = re.sub(r"[_\-.]+", " ", n)
    n = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n)      # CamelCase -> Camel Case
    return re.sub(r"\s+", " ", n).strip().lower()


def informative(names):
    """False if folder names carry no semantics (e.g. '0'..'15' or WordNet ids)."""
    bad = [re.fullmatch(r"(n\d{8}|\d+|class\s?\d+)", n.strip().lower()) for n in names]
    return not any(bad)


def load_classnames(classes, path=None):
    """Optional override file: one line per class, either 'folder<TAB>name' or just names in folder order."""
    if not path:
        return [clean_name(c) for c in classes]
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    if all("\t" in l or ":" in l for l in lines):
        m = {}
        for l in lines:
            k, v = re.split(r"\t|:", l, maxsplit=1)
            m[k.strip()] = v.strip()
        return [m.get(c, clean_name(c)) for c in classes]
    assert len(lines) == len(classes), "classnames file must have one line per class"
    return [l.strip() for l in lines]


def prompts_for(name, templates=TEMPLATES, extra=None):
    ps = [t.format(name) for t in templates]
    if extra:
        ps += [t.format(name) for t in extra]
    return ps
