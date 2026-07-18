# SCMG

**A multi-omics and stage-resolved single-cell framework for pan-cancer driver-gene prioritization**

SCMG is a graph-learning framework that integrates three bulk omics views, stage- and source-resolved epithelial single-cell RNA-seq information, and gene interaction networks.

The released code implements five-fold training and evaluation across six gene interaction networks.

---

## Model overview

[![SCMG model overview](overview.png)](overview.png)

SCMG contains three main components:

### 1. Disagreement-aware multi-view bulk encoder

The bulk branch uses three 16-dimensional omics views:

- somatic mutation;
- DNA methylation;
- gene expression.

Each omics view is processed using local and graph-based encoders. Information from the three views is subsequently combined to produce a bulk driver-gene logit.

### 2. Stage-source scRNA encoder

The single-cell branch uses epithelial cells representing three disease states:

- **Normal**
- **Precancer**
- **Cancer**

For each gene, the released code constructs stage- and source-resolved pseudobulk tokens from the complete single-cell matrices. These tokens summarize:

- mean log-normalized expression;
- expressing-cell proportion;
- mean expression among expressing cells;
- raw expression per cell.

The code additionally constructs a 13-dimensional stage-summary representation and models both stage states and progression differences.

The released scRNA prediction is obtained by averaging the logits from:

- a stage-source Transformer encoder;
- a hybrid progression encoder.

### 3. Late-logit fusion

The bulk and scRNA branches are trained separately. Their logits are subsequently combined using a trainable linear late-logit mixer.

Fold-specific scRNA normalization and model fitting use training-fold genes only.

---

## Repository structure

```text
SCMG/
├── main.py               # Five-fold training and evaluation
├── model.py              # Bulk encoder, scRNA encoders and logit mixer
├── utils.py              # HDF5 loading, fold construction and evaluation
├── requirements.txt      # Reproducibility environment
├── overview.png     # Model overview
└── PPI/
    ├── README.md
    ├── CPDB_multiomics.h5
    ├── STRING_multiomics.h5
    ├── IRef_multiomics.h5
    ├── PCNet_multiomics.h5
    ├── PATHNET_multiomics.h5
    └── GGNET_multiomics.h5
```

The six HDF5 files are not stored directly in the GitHub repository because of their file sizes.

---

## Data download

### Model-ready HDF5 files

Download the six model-ready datasets from Google Drive:

### [Download SCMG model-ready data](https://drive.google.com/drive/folders/1m6iU56zQgVlk6xdb2G2c9AW1EZIth4Tm?usp=drive_link)

After downloading, place all six files in the existing `PPI/` directory:

```text
SCMG/PPI/
```

The filenames must remain exactly as follows:

| Network | Required filename |
|---|---|
| CPDB | `CPDB_multiomics.h5` |
| STRING | `STRING_multiomics.h5` |
| IRefIndex | `IRef_multiomics.h5` |
| PCNet | `PCNet_multiomics.h5` |
| PathNet | `PATHNET_multiomics.h5` |
| GGNet | `GGNET_multiomics.h5` |

Each model-ready HDF5 container includes the corresponding gene interaction network, bulk multi-omics features, driver-gene labels, single-cell expression matrices and source metadata required by the released training code.

> **Important:** the commands below explicitly use `--data-path PPI` because the downloadable files are placed in the repository-level `PPI/` directory.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mantouwang/SCMG.git
cd SCMG
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The environment used for the released experiment included:

- Python 3.9.23
- CUDA 12.8
- PyTorch 2.9.0 nightly build
- PyTorch Geometric 2.6.1
- NumPy 1.26.4
- h5py 3.14.0
- scikit-learn 1.4.2

A CUDA-enabled GPU is recommended. The script falls back to CPU execution when CUDA is unavailable.

---

## Running SCMG

### CPDB

```bash
python main.py CPDB --data-path PPI --device cuda:0
```

### Other interaction networks

```bash
python main.py STRING --data-path PPI --device cuda:0
python main.py IREF --data-path PPI --device cuda:0
python main.py PCNET --data-path PPI --device cuda:0
python main.py PATHNET --data-path PPI --device cuda:0
python main.py GGNET --data-path PPI --device cuda:0
```

To run on CPU:

```bash
python main.py CPDB --data-path PPI --device cpu
```

The program reports the mean area under the precision–recall curve across five stratified folds:

```text
Mean Final AUPR: ...
```

---

## Training workflow

For every cross-validation fold, the released script performs the following steps:

1. Construct stage-source single-cell summaries from the complete cell matrices.
2. Normalize the single-cell features using training-fold genes.
3. Train the bulk branch.
4. Train the scRNA branch.
5. Freeze the modality-specific branches and fit the late-logit mixer.
6. Evaluate the final fused prediction on the held-out fold.

The current released configuration is defined directly in `main.py`.

| Component | Epochs | Learning rate |
|---|---:|---:|
| Bulk branch | 1200 | `8 × 10⁻⁴` |
| scRNA branch | 140 | `2.4 × 10⁻⁴` |
| Late-logit mixer | 100 | `1 × 10⁻³` |

The random seed is set to 42, and gradient clipping is applied with a maximum norm of 5.

---

## Supported gene interaction networks

SCMG can be evaluated using the following six network backbones:

- CPDB
- STRING
- IRefIndex
- PCNet
- PathNet
- GGNet

The network name can be supplied either as a positional argument:

```bash
python main.py CPDB --data-path PPI
```

or through the `--dataset` option:

```bash
python main.py --dataset CPDB --data-path PPI
```

When no network is specified, the script uses CPDB.

---

## Data provenance

The model-ready HDF5 files contain processed inputs used by the released SCMG training workflow. The original datasets remain available from their respective public databases.

<details>
<summary><strong>Bulk multi-omics and single-cell datasets</strong></summary>

<br>

### Bulk multi-omics data

Bulk somatic mutation, DNA methylation and gene-expression features were derived from 16 TCGA cancer cohorts:

`BLCA`, `BRCA`, `CESC`, `COAD`, `ESCA`, `HNSC`, `KIRC`, `KIRP`, `LIHC`, `LUAD`, `LUSC`, `PRAD`, `READ`, `STAD`, `THCA` and `UCEC`.

### Single-cell RNA-seq datasets

The stage-resolved epithelial single-cell data were obtained from:

- [GSE161277](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161277)
- [GSE218170](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218170)
- [GSE251990](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251990)

The processed inputs represent epithelial cells from Normal, Precancer and Cancer states.

</details>

<details>
<summary><strong>Driver-gene labels and candidate-background resources</strong></summary>

<br>

| Resource | Version or role | Access |
|---|---|---|
| Network of Cancer Genes | NCG version 6.0 | [NCG](http://ncg.kcl.ac.uk/cancer_genes.php) |
| DigSEE | Literature-mined cancer-gene resource | [DigSEE](http://210.107.182.61/digseeOld/) |
| COSMIC Cancer Gene Census | CGC version 91 | [COSMIC downloads](https://cancer.sanger.ac.uk/cosmic/download) |
| OMIM | Cancer-gene candidate-background curation | [OMIM downloads](https://omim.org/downloads) |
| KEGG Pathways in Cancer | Cancer-pathway gene collection | [MSigDB entry](https://www.gsea-msigdb.org/gsea/msigdb/cards/KEGG_PATHWAYS_IN_CANCER.html) |

The exact label-construction and candidate-background rules are described in the manuscript.

</details>

<details>
<summary><strong>External validation and downstream resources</strong></summary>

<br>

| Resource | Use in the study | Access |
|---|---|---|
| IntOGen | External driver-catalogue analysis | [IntOGen](https://www.intogen.org/search) |
| UCSC Xena | TCGA expression, clinical and survival data | [UCSC Xena](https://xenabrowser.net/datapages/) |
| MSigDB human v2025.1 | Hallmark, KEGG Legacy and GO BP gene sets | [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb) |

The downstream biological and patient-level analyses are described in the manuscript. Their complete analysis scripts are not included in the current core training release.

</details>

---

## Reproducibility scope

This repository provides the core SCMG implementation and the entry point used to train and evaluate the released model across six interaction networks.

It includes:

- bulk and scRNA model definitions;
- stage-source single-cell feature extraction;
- five-fold stratified evaluation;
- fold-specific single-cell normalization;
- sequential branch and fusion training;
- final AUPR calculation.

The repository does not currently include every script used for downstream pathway enrichment, epithelial evolutionary-shift analysis, external catalogue analysis or patient-level outcome analysis.

---

## Citation

Citation information will be added after publication.

---




