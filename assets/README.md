# Datasets and checkpoints

Processed datasets and pretrained weights are **not** shipped in this repo. Download them from the upstream sources below and place them under `assets/download/` using the exact relative paths listed here.

Optional check after placement:

```bash
mol-cache assets validate --model <model> --dataset <dataset>
```

## Expected layout

```text
assets/download/
├── checkpoints/
│   ├── semlaflow/geom/200epochs.ckpt
│   ├── tabasco/geom/tabasco-geom-hot.ckpt
│   ├── flowr/spindr/flowr.ckpt
│   └── flowr_root/flowr_root_v2.1.ckpt
└── data/
    ├── semlaflow/geom/
    │   ├── test.smol
    │   └── train.smiles
    ├── tabasco/geom/
    │   ├── processed_geom_train.pt
    │   ├── processed_geom_val.pt
    │   └── processed_geom_test.pt
    ├── flowr/spindr/
    │   ├── train.smol
    │   ├── val.smol
    │   └── test.smol
    └── flowr_root/crossdocked/
        └── test.smol
```


## SemlaFlow × GEOM

| File | Place at |
|------|----------|
| GEOM checkpoint (`200epochs.ckpt` or equivalent) | `assets/download/checkpoints/semlaflow/geom/200epochs.ckpt` |
| Test split (`.smol`) | `assets/download/data/semlaflow/geom/test.smol` |
| Train SMILES (novelty metric) | `assets/download/data/semlaflow/geom/train.smiles` |

**Upstream:** [SemlaFlow Google Drive checkpoints](https://drive.google.com/drive/folders/1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT?usp=sharing) (see also `models/semlaflow/README.md`). Processed GEOM `.smol` / `train.smiles` come from the SemlaFlow data prep pipeline described there.

---

## Tabasco × GEOM

| File | Place at |
|------|----------|
| `tabasco-geom-hot.ckpt` | `assets/download/checkpoints/tabasco/geom/tabasco-geom-hot.ckpt` |
| `processed_geom_{train,val,test}.pt` | `assets/download/data/tabasco/geom/` |

**Upstream (Hugging Face):**

- Checkpoint: [carlosinator/tabasco-geom-hot](https://huggingface.co/carlosinator/tabasco-geom-hot)
- Data: [carlosinator/tabasco-geom-drugs](https://huggingface.co/datasets/carlosinator/tabasco-geom-drugs)

---

## FLOWR × SPINDR

| File | Place at |
|------|----------|
| `flowr.ckpt` | `assets/download/checkpoints/flowr/spindr/flowr.ckpt` |
| `train.smol`, `val.smol`, `test.smol` | `assets/download/data/flowr/spindr/` |

**Upstream (Zenodo):**

- Checkpoint: [zenodo.org/records/15737419](https://zenodo.org/records/15737419)
- Processed SPINDR `.smol` splits: [zenodo.org/records/15257565](https://zenodo.org/records/15257565)

Use the non-explicit-hydrogen checkpoint.

---

## FLOWR.root × SPINDR

| File | Place at |
|------|----------|
| `flowr_root_v2.1.ckpt` | `assets/download/checkpoints/flowr_root/flowr_root_v2.1.ckpt` |
| SPINDR `.smol` splits | Same as FLOWR: `assets/download/data/flowr/spindr/` |

**Upstream:** [FLOWR.root Google Drive](https://drive.google.com/drive/u/0/folders/1NWpzTY-BG_9C4zXZndWlKwdu7UJNCYj8) (checkpoint v2.1).

---

## FLOWR.root × CrossDocked

| File | Place at |
|------|----------|
| `flowr_root_v2.1.ckpt` | `assets/download/checkpoints/flowr_root/flowr_root_v2.1.ckpt` (same file as SPINDR) |
| `test.smol` | `assets/download/data/flowr_root/crossdocked/test.smol` |

**Upstream:** [FLOWR.root Google Drive](https://drive.google.com/drive/u/0/folders/1NWpzTY-BG_9C4zXZndWlKwdu7UJNCYj8) (checkpoint v2.1).

