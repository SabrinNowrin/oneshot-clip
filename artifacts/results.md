Eval set: test (400 images), 16 classes, 1 shot/class, backbones: ViT-SO400M-16-SigLIP2-384:webli, ViT-H-14-quickgelu:dfn5b

Hyper-parameters: {'beta': 8.0, 'cache_w': 0.5, 'steps': 300, 'lr': 0.005, 'anchor': 2.0, 'tr_iters': 10, 'tr_prior_w': 1.0, 'tr_shot_w': 0.5}

| Method | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| ViT-SO400M-16-SigLIP2-384 | zero-shot text | 90.00 | 90.00 | 90.00 | **90.00 ± 0.00** |
| ViT-SO400M-16-SigLIP2-384 | 1-shot prototype (raw image) | 73.50 | 72.50 | 76.75 | **74.25 ± 1.81** |
| ViT-SO400M-16-SigLIP2-384 | 1-shot prototype (augmented) | 77.25 | 79.75 | 80.75 | **79.25 ± 1.47** |
| ViT-SO400M-16-SigLIP2-384 | view cache | 78.00 | 80.25 | 80.75 | **79.67 ± 1.20** |
| ViT-SO400M-16-SigLIP2-384 | anchored head | 91.50 | 92.25 | 91.75 | **91.83 ± 0.31** |
| ViT-SO400M-16-SigLIP2-384 | head + cache | 88.75 | 88.00 | 88.00 | **88.25 ± 0.35** |
| ViT-H-14-quickgelu | zero-shot text | 88.75 | 88.75 | 88.75 | **88.75 ± 0.00** |
| ViT-H-14-quickgelu | 1-shot prototype (raw image) | 76.75 | 84.75 | 80.00 | **80.50 ± 3.29** |
| ViT-H-14-quickgelu | 1-shot prototype (augmented) | 83.00 | 85.25 | 81.75 | **83.33 ± 1.45** |
| ViT-H-14-quickgelu | view cache | 83.75 | 85.25 | 82.00 | **83.67 ± 1.33** |
| ViT-H-14-quickgelu | anchored head | 87.25 | 89.50 | 89.75 | **88.83 ± 1.12** |
| ViT-H-14-quickgelu | head + cache | 90.25 | 92.75 | 88.00 | **90.33 ± 1.94** |
| ENSEMBLE | inductive | 91.00 | 91.50 | 87.75 | **90.08 ± 1.66** |
| ENSEMBLE | + transductive (softmax) | 94.50 | 93.00 | 86.75 | **91.42 ± 3.36** |
| ENSEMBLE | + transductive (balanced) | 91.50 | 93.75 | 89.00 | **91.42 ± 1.94** |
