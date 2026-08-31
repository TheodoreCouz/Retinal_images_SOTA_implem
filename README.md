# Retinal Images — State-of-the-Art Implementations

Reference implementations of the three published methods used as baselines for
multi-label retinal disease classification on **MuReD**, **RFMiD** and **PRISM**.

Each method lives in its own directory, keeping the upstream layout so the code
stays recognisable against the original repository, with the fundus-specific
training and evaluation scripts added alongside.

| Method | Backbone | Origin |
|---|---|---|
| [C-Tran](#c-tran) | DenseNet-161 / RETFound + label-query transformer | [QData/C-Tran](https://github.com/QData/C-Tran) |
| [Pytorch-RIADD](#pytorch-riadd) | EfficientNet-B6 (NoisyStudent) via `timm` | [Hanson0910/Pytorch-RIADD](https://github.com/Hanson0910/Pytorch-RIADD) |
| [RetExpert](#retexpert) | RETFound ViT-L + Knowledge-Unit adapters | [OVS-AILab/RetExpert](https://github.com/OVS-AILab/RetExpert) |

---

## Repository layout

```
Retinal_images_SOTA_implem/
├── C-Tran/           label-query transformer for multi-label classification
├── Pytorch-RIADD/    RIADD (ISBI-2021) challenge solution, vendored timm fork
└── RetExpert/        adapter-based foundation-model framework with test-time adaptation
```

Only **source code** is tracked. Datasets, pretrained weights, checkpoints and
run outputs are excluded — see [Datasets and weights](#datasets-and-weights).

---

## C-Tran

**Paper.** Jack Lanchantin, Tianlu Wang, Vicente Ordóñez Román, Yanjun Qi.
*General Multi-label Image Classification with Transformers.* CVPR 2021.
[arXiv:2011.14027](https://arxiv.org/abs/2011.14027)

```bibtex
@article{lanchantin2020general,
  title={General Multi-label Image Classification with Transformers},
  author={Lanchantin, Jack and Wang, Tianlu and Ordonez, Vicente and Qi, Yanjun},
  journal={arXiv preprint arXiv:2011.14027},
  year={2020}
}
```

**Source.** https://github.com/QData/C-Tran

**Method.** Treats each label as a query token. A transformer over image features
and label embeddings, trained with Label Mask Training (LMT), learns label
co-occurrence directly instead of predicting each class independently — a good
fit for retinal images where several signs appear together.

### Files

*Fundus experiments (added here):*
```
train_fundus.py                 train on MuReD / RFMiD / PRISM
train_mured.py                  train on MuReD (shared transforms + epoch loop)
dataloaders/fundus_prep.py      split preparation for the three datasets
dataloaders/mured_dataset.py    dataset wrapper
dataloaders/resampling.py       LP-ROS resampling for the long tail
utils/{retexpert,mured,fiber}_metrics.py   evaluation protocols
eval_performance_fundus.py, eval_performance_mured.py, summarize_ci.py,
eval_best_threshold_ci.py, mured_paper_ci.py, report_mured.py,
write_performance_reports.py, run_ci_queue.sh
RETFound/                       C-Tran variant using a RETFound ViT-L backbone
```

*Upstream (unmodified):* `models/` (CTran, backbone, position_enc,
transformer_layers), `config_args.py`, `optim_schedule.py`, `run_epoch.py`,
`utils/{evaluate,logger,metrics}.py`, plus `main.py`, `load_data.py` and the
COCO / VOC / VG / NUS / CUB dataloaders, which drive the paper's original
benchmarks and are **not** used by the fundus path. They are kept because
`models/__init__.py` imports `CTran_cub`, so removing them would mean editing
upstream files.

### Train

```bash
python train_fundus.py --dataset prism \
    --backbone densenet161 --img_size 384 \
    --batch_size 16 --grad_ac_steps 2 --lr 1e-5 \
    --name ctran_poly_constlr_bs32 --results_dir results
```
`--dataset` accepts `mured`, `rfmid` or `prism`.

---

## Pytorch-RIADD

**Source.** https://github.com/Hanson0910/Pytorch-RIADD — the author's solution
to the **RIADD (ISBI-2021)** Retinal Image Analysis for multi-Disease Detection
challenge, hosted at https://riadd.grand-challenge.org/.

**Built on `timm`.** The repository is a fork of
[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
(Ross Wightman). The fork is vendored under `timm/` (141 modules) because the
training scripts import it directly and it carries the author's modifications;
installing `timm` from PyPI is **not** equivalent.

```bibtex
@misc{rwightman_timm,
  author = {Ross Wightman},
  title  = {PyTorch Image Models},
  year   = {2019},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

**Method.** EfficientNet-B6 (NoisyStudent weights) trained with heavy
augmentation and K-fold cross-validation; the challenge submission averages the
folds. Reported here both as a single fold and as the 5-fold ensemble.

### Files

*Fundus experiments (added here):*
```
train_prism_kfold_ddp.py    train_prism_eb6_ddp.py      PRISM
train_mured_kfold_ddp.py    train_mured_eb6_ddp.py      MuReD
train_riadd_kfold_ddp.py    train_riadd_eb6_ddp.py      RFMiD
eval_prism.py  eval_mured.py  eval_mured_strat.py
eval_ci.py  eval_ensemble.py  eval_all_metrics.py  eval_metrics9.py
metrics7.py  metrics9.py  run_ci_ensembles.sh  relaunch_mured_strat.sh
```

*Upstream:* `timm/`, plus `inference.py`, `hubconf.py`, `avg_checkpoints.py`,
`clean_checkpoint.py`, `sotabench*`, `tests/`, `setup.py`.

### Train

```bash
python -m torch.distributed.launch --nproc_per_node=2 \
    train_prism_kfold_ddp.py \
    --data '/path/to/PRISM/train' \
    --model tf_efficientnet_b6_ns \
    --output ckpt/tf_efficientnet_b6_ns-PRISM-KFOLD-SGD1E-1
```

> **Note.** The `train_*_ddp.py` scripts carry absolute dataset paths as argparse
> defaults (e.g. `/home/cousin/research/...`). Pass `--data` and `--output`
> explicitly, or edit the defaults, before running elsewhere.

---

## RetExpert

**Paper.** Hongyang Jiang, Zirong Liu, Mengdi Gao, et al. *RetExpert: A test-time
clinically adaptive framework for detecting multiple fundus diseases by
harnessing ophthalmic foundation models.* 2025.

```bibtex
@article{jiang2025retexpert,
  title={RetExpert: A test-time clinically adaptive framework for detecting
         multiple fundus diseases by harnessing ophthalmic foundation models},
  author={Jiang, Hongyang and Liu, Zirong and Gao, Mengdi and others},
  year={2025}
}
```
*(The upstream README's BibTeX leaves the journal field as a placeholder;
check the publication record before citing.)*

**Source.** https://github.com/OVS-AILab/RetExpert (official implementation, MIT)

**Method.** Parameter-efficient adaptation of a RETFound ViT-L through Knowledge
Unit (KU) adapters, combined with a Fundus Disease Co-occurrence Matrix to
suppress medically contradictory predictions, uncertainty-aware multi-label
learning for the long tail, and two-stage test-time adaptation for domain shift.

### Files

```
train.py                    training entry point
test_tta.py                 test-time adaptation
config.py                   per-dataset configuration, trainable-parameter selection
models/vit_adapter.py       RETFound ViT-L + KU adapters
losses/retexpert_loss.py    uncertainty-aware multi-label loss
utils/                      datasets, engine, LR schedules/decay, misc
scripts/                    launch scripts, incl. the multi-seed runs used here
```

Multi-seed launchers added for the comparison:
`run_mured_5seeds.sh`, `run_mured_5seeds_400ep.sh`, `run_mured_4seeds_400ep.sh`,
`run_rfmid_4seeds_200ep.sh`, `run_rfmid_5seeds_400ep.sh`,
`run_prism_5seeds_200ep.sh`, plus `compute_recall_at_5.py` and
`evaluate_new_metrics.py`.

### Train

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
    train.py \
    --dataset MuReD --data_path /path/to/MuReD --nb_classes 20 \
    --model vit_large_patch16 --input_size 224 --batch_size 16 \
    --finetune /path/to/RETFound_cfp_weights.pth
```
See `scripts/run_train.sh` for the full argument list and `scripts/run_tta_ADAM.sh`
for test-time adaptation.

---

## Datasets and weights

Not included, and not redistributable here:

- **MuReD** — Multi-label Retinal Diseases dataset
- **RFMiD** — Retinal Fundus Multi-Disease Image Dataset
  ([RIADD challenge](https://riadd.grand-challenge.org/))
- **PRISM** — internal dataset
- **RETFound** pretrained weights — required by RetExpert and the C-Tran/RETFound
  variant; obtain from the [RETFound repository](https://github.com/rmaphoh/RETFound_MAE)

Point each script at your local copies via its `--data` / `--data_path` /
`--finetune` arguments.

## Environments

Each directory keeps its own `requirements.txt`; the three methods have
incompatible dependency sets (notably `timm` — Pytorch-RIADD needs its vendored
fork, not the PyPI package). Use a separate virtual environment per method.

## Provenance and licensing

Each subdirectory is a copy of the upstream repository's source with
fundus-specific training and evaluation code added. Upstream licences and
copyright continue to apply to the original files — consult each source
repository. This repository exists to make the baseline experiments
reproducible; please cite the original papers above, not this repository.
