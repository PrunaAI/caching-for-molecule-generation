# Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation
[![arXiv](https://img.shields.io/badge/arXiv-2510.04646-b31b1b.svg)](https://arxiv.org/abs/2510.04646)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-blue)](https://openreview.net/forum?id=NaLVutCHCI)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red)]()

Flow matching models generate high-fidelity molecular geometries but incur significant computational costs during inference, requiring hundreds of neural network evaluations. This inference cost becomes the primary bottleneck when such models are employed in practice to sample large numbers of molecular candidates. This work presents a training-free caching strategy that accelerates molecular geometry generation by predicting intermediate hidden states across solver steps. This caching scheme operates directly on the SE(3)-equivariant backbone, is compatible with pretrained models, and is orthogonal to existing training-based accelerations and system-level optimizations. Experiments on molecular geometry generation demonstrate that caching achieves a twofold reduction in wall-clock inference time at matched sample quality and a speedup of up to 3× with minimal sample quality degradation. Because these gains compound with other optimizations, applying caching alongside other general, lossless optimizations yield as much as a 7× speedup.

**Licensing**: Apache-2.0 for orchestration code (`LICENSE`); third-party model code retains upstream MIT licenses (`THIRD_PARTY_NOTICES.md`).

## Quick start

```bash
# 1. Create the sampling environment
conda env create -f environment.yml
conda activate mol-cache

# 2. List supported routes
mol-cache list

# 3. Sample in-process (GPU). All routes share one load → smash → sample path.
mol-cache sample --model semlaflow --dataset geom --n-samples 100 --cache-interval 2 --cache-mode taylor --device 0
```

Supported routes: `semlaflow@geom`, `tabasco@geom`, `flowr@spindr`,
`flowr_root@spindr`, `flowr_root@crossdocked`. Sampling writes
`outputs/<model>_<dataset>_<timestamp>/` with `resolved_config.json` and
`molecules.sdf`.

## Overview 

`mol-cache sample` keeps the familiar `--model` / `--dataset` flags and
composes a package-shipped Hydra config under `mol_cache/conf/`. Each model
file (`model/semlaflow.yaml`, …) owns defaults; only explicitly supplied CLI
flags override them.

Concrete adapters live beside each vendored package and implement
`load()`, `sample()`, and `configure_cache()` on `mol_cache.model.MolCacheModel`.
The thin orchestrator in `mol_cache/sample.py` then runs the shared lifecycle:

`validate → load → smash (optional) → configure_cache → sample → summary`

Vendored FLOWR packages are named `flowr_model` and `flowr_root_model` so both
can load in the same interpreter.

Since configuration and application of caching works by registering the algorithm with the `pruna` package, the optimization config can easily be extended to include [further optimization](https://docs.pruna.ai/en/stable/compression.html) algorithms available in the `pruna` package. Find more information [in the documentation](https://docs.pruna.ai/en/stable/setup/index.html).

## Environment

`environment.yml` — unified sampling stack (Python 3.11, PyTorch 2.7.1, `pruna==0.3.4`).

## Assets

Follow `assets/README.md` to download all necessary weights and data.

## Citation

Please cite the original model papers (see `assets/provenance.yaml`) and this caching work when using the repository.
```bibtex
@article{sommer2026predictive,
  title = {Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation},
  author = {Sommer, Johanna and Fleischmann, Nils and Rachwan, John and G{\"u}nnemann, Stephan and Charpentier, Bertrand},
  journal = {Transactions on Machine Learning Research},
  issn = {2835-8856},
  year = {2026},
  url = {https://openreview.net/forum?id=NaLVutCHCI}
}
```
