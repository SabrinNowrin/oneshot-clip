"""Frozen foundation-model encoders (any open_clip model: CLIP, OpenCLIP, DFN, EVA, SigLIP, SigLIP2...)."""
import os

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .text import prompts_for


def parse_spec(spec):
    name, _, pre = spec.partition(":")
    return name, (None if pre in ("", "none") else pre)


def safe(spec):
    return spec.replace("/", "_").replace(":", "__")


class _Views(Dataset):
    def __init__(self, paths, tfm, n_views, clean_tfm=None):
        self.paths, self.tfm, self.n, self.clean = paths, tfm, n_views, clean_tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        views = [self.clean(img)] if self.clean is not None else []
        views += [self.tfm(img) for _ in range(self.n)]
        return torch.stack(views)


class Backbone:
    def __init__(self, spec, device):
        self.spec = spec
        self.device = device
        name, pre = parse_spec(spec)
        prec = "fp32"
        if pre is None:  # random-init models (smoke tests only): make them reproducible
            torch.manual_seed(0)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pre, device=device, precision=prec)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)
        v = self.model.visual
        size = v.image_size if hasattr(v, "image_size") else 224
        self.size = size[0] if isinstance(size, (tuple, list)) else size
        mean = getattr(v, "image_mean", None) or open_clip.OPENAI_DATASET_MEAN
        std = getattr(v, "image_std", None) or open_clip.OPENAI_DATASET_STD
        norm = T.Normalize(mean, std)
        bic = T.InterpolationMode.BICUBIC
        # test-time: the model's own preprocessing + horizontal flip
        self.flip = T.Compose([T.RandomHorizontalFlip(p=1.0), self.preprocess])
        # one-shot support augmentation: many plausible "re-photographs" of the single image
        self.augment = T.Compose([
            T.RandomResizedCrop(self.size, scale=(0.35, 1.0), interpolation=bic),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.25, 0.04)], p=0.6),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(), norm,
        ])

    @torch.no_grad()
    def _encode(self, paths, tfm, n_views, clean, bs, workers):
        dl = DataLoader(_Views(paths, tfm, n_views, clean), batch_size=bs, num_workers=workers)
        feats = []
        for x in dl:
            b, v = x.shape[:2]
            x = x.flatten(0, 1).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                f = F.normalize(self.model.encode_image(x).float(), dim=-1)
            feats.append(f.view(b, v, -1).cpu())
        return torch.cat(feats)  # [N, V, D]

    def encode_test(self, paths, bs=64, workers=2, tta=True):
        """[N, D]: mean of (clean, h-flip) view embeddings, re-normalised."""
        if tta:
            f = self._encode(paths, self.flip, 1, self.preprocess, bs, workers)
        else:
            f = self._encode(paths, None, 0, self.preprocess, bs, workers)
        return F.normalize(f.mean(1), dim=-1)

    def encode_support(self, paths, n_aug=64, workers=2, seed=0):
        """[N, 1 + n_aug, D]: view 0 is the clean image, the rest are augmentations."""
        torch.manual_seed(seed)
        bs = max(1, 256 // (n_aug + 1))
        return self._encode(paths, self.augment, n_aug, self.preprocess, bs, 0)

    @torch.no_grad()
    def encode_text(self, classnames, templates=None):
        ws = []
        for n in classnames:
            toks = self.tokenizer(prompts_for(n) if templates is None else [t.format(n) for t in templates])
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                e = F.normalize(self.model.encode_text(toks.to(self.device)).float(), dim=-1)
            ws.append(F.normalize(e.mean(0), dim=-1))
        return torch.stack(ws).cpu()  # [C, D]

    def free(self):
        del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


def cached(path, fn):
    if os.path.exists(path):
        return torch.load(path)
    out = fn()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    return out
