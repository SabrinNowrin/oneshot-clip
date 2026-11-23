"""Glue: build the per-backbone one-shot classifier and run (inductive / transductive) prediction."""
import torch
import torch.nn.functional as F

from . import methods as M

DEFAULTS = dict(beta=8.0, cache_w=0.5, steps=300, lr=5e-3, anchor=2.0,
                tr_iters=10, tr_prior_w=1.0, tr_shot_w=0.5)


def fit_backbone(S, Wt, hp, seed):
    """S: [C, V, D] support views, Wt: [C, D] text weights or None."""
    proto = F.normalize(S.mean(1), dim=-1)
    W0 = F.normalize(Wt + proto, dim=-1) if Wt is not None else proto
    W = M.train_head(S, W0, steps=hp["steps"], lr=hp["lr"], anchor=hp["anchor"], seed=seed)
    return dict(W=W, S=S, proto=proto, Wt=Wt)


def backbone_logits(X, m, hp):
    return M.head_logits(X, m["W"]) + hp["cache_w"] * M.view_cache(X, m["S"], hp["beta"])


def predict(Xs, models, hp, transductive=False, balanced=True):
    """Xs: list of [N, D] test embeddings (one per backbone). Returns [N, C] log-scores."""
    L0 = M.ensemble([backbone_logits(X, m, hp) for X, m in zip(Xs, models)])
    if not transductive:
        return L0
    return M.transductive(Xs, [m["proto"] for m in models], L0, iters=hp["tr_iters"],
                          prior_w=hp["tr_prior_w"], shot_w=hp["tr_shot_w"], balanced=balanced)
