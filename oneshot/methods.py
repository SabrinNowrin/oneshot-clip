"""One-shot classifiers built on frozen embeddings.

Notation: C classes, V support views per class (1 clean + augmentations), D dims, N test images.
All embeddings are L2-normalised; logits are `SCALE * cosine` so every branch lives on the same scale.
"""
import torch
import torch.nn.functional as F

SCALE = 100.0


def zero_shot(X, Wt):
    """Text-only classifier (prompt ensemble)."""
    return SCALE * X @ Wt.T


def prototype(X, S, clean_only=False):
    """Nearest class prototype: the shot itself, or the mean of its augmented views."""
    P = S[:, 0] if clean_only else S.mean(1)
    return SCALE * X @ F.normalize(P, dim=-1).T


def view_cache(X, S, beta=8.0):
    """Tip-Adapter-style key/value cache over every augmented view.

    Per class: soft-max (log-mean-exp, sharpness beta) of cosine to its V views. This is a smooth
    'nearest augmented view' score, so a test image that matches one crop of the shot well scores
    high even if the global prototype is off.
    """
    C, V, D = S.shape
    sim = X @ S.reshape(C * V, D).T                                    # [N, C*V]
    sim = sim.view(-1, C, V)
    lme = (torch.logsumexp(beta * sim, dim=-1) - torch.log(torch.tensor(float(V)))) / beta
    return SCALE * lme


def train_head(S, W0, steps=300, lr=5e-3, anchor=2.0, label_smooth=0.1, seed=0, mixup=True):
    """Cosine linear head trained on the augmented views of the single shots.

    W is initialised at W0 (text + prototype fusion) and pulled back toward it with an L2 anchor
    (as in CLAP, Silva-Rodriguez et al. 2024), so 16 images can only *refine* the foundation
    model's prior rather than overwrite it. Feature-level mixup between views of different
    classes regularises the decision boundaries further.
    """
    g = torch.Generator().manual_seed(seed)
    C, V, D = S.shape
    x = S.reshape(C * V, D)
    y = torch.arange(C).repeat_interleave(V)
    W = torch.nn.Parameter(W0.clone())
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Y = F.one_hot(y, C).float() * (1 - label_smooth) + label_smooth / C
    for _ in range(steps):
        xb, Yb = x, Y
        if mixup:
            perm = torch.randperm(len(x), generator=g)
            lam = torch.distributions.Beta(0.4, 0.4).sample((len(x), 1))
            lam = torch.maximum(lam, 1 - lam)                          # keep the dominant label
            xb = F.normalize(lam * x + (1 - lam) * x[perm], dim=-1)
            Yb = lam * Y + (1 - lam) * Y[perm]
        logits = SCALE * xb @ F.normalize(W, dim=-1).T
        loss = -(Yb * logits.log_softmax(-1)).sum(-1).mean()
        loss = loss + anchor * ((W - W0) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return F.normalize(W.detach(), dim=-1)


def head_logits(X, W):
    return SCALE * X @ W.T


def sinkhorn(L, iters=3):
    """Soft assignment with a uniform class marginal (each class gets ~N/C test images)."""
    Q = L.softmax(-1)
    N, C = Q.shape
    for _ in range(iters):
        Q = Q * ((N / C) / Q.sum(0, keepdim=True).clamp_min(1e-8))
        Q = Q / Q.sum(1, keepdim=True)
    return Q


def transductive(Xs, protos, L0, iters=10, prior_w=1.0, shot_w=0.5, balanced=True):
    """Test-time adaptation on the unlabelled test set (no labels used).

    Soft k-means in each backbone's embedding space, initialised from the inductive prediction L0:
      1. Q = assignment of test images to classes (Sinkhorn-balanced or plain softmax)
      2. per backbone, class centroid = normalise(normalise(Q^T X) + shot_w * shot_prototype)
      3. new logits = mean over backbones of cosine to centroids + prior_w * L0
    The single labelled shot keeps each centroid anchored to its class; the prior term stops drift.
    """
    assign = sinkhorn if balanced else (lambda L: L.softmax(-1))
    Q = assign(L0)
    for _ in range(iters):
        L = 0
        for X, P in zip(Xs, protos):
            mu = F.normalize(Q.T @ X, dim=-1) + shot_w * F.normalize(P, dim=-1)
            L = L + SCALE * X @ F.normalize(mu, dim=-1).T
        L = L / len(Xs) + prior_w * L0
        Q = assign(L)
    return Q.log()


def ensemble(logit_list):
    """Average log-probabilities across backbones."""
    return torch.stack([l.log_softmax(-1) for l in logit_list]).mean(0)


def acc(L, y):
    return (L.argmax(-1) == y).float().mean().item() * 100
