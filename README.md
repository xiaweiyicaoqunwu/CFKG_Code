# CFKG: Citrus Flavor Knowledge Graph

Research code for **A multi-source knowledge graph of citrus sensory properties**.

CFKG integrates knowledge about citrus germplasm, compounds, sensory properties, production areas, and metabolic processes from 16 public databases. This repository provides code for source data processing, LLM-assisted entity alignment, and knowledge graph embedding experiments.

The final graph contains **9,602 object entities and 62,433 object triples** across eight entity types and ten relation types. Including attribute entities and triples, CFKG contains 164,875 entities and 376,984 triples.

The ontology and final graph exports are available on [Figshare](https://doi.org/10.6084/m9.figshare.33069161).

## Repository structure

```text
CFKG_Code/
├── raw_data_processing/       # Convert cleaned XLSX tables and import into Neo4j
│   ├── xlsx2json.py
│   └── json2neo4j.py
├── entity_alignment/          # Generate candidates, align entities, and evaluate
│   ├── config.py
│   ├── data_extraction.py
│   ├── embedding_generation.py
│   ├── candidate_generation.py
│   ├── llm_alignment.py
│   ├── evaluate_alignment.py
│   ├── evaluate_alignment_by_type.py
│   └── data/                  # Results, test pairs, and merge_entities.py
└── link_prediction/           # Train and evaluate five embedding models
    ├── data_loader.py
    ├── baseline_models.py
    ├── train_baseline.py
    ├── evaluate_baseline.py
    └── data/                  # dataset.pkl
```

## Installation

Use **Python 3.10**. The following commands provide a suggested environment; the original experiments do not have a dependency lockfile.

```bash
git clone https://github.com/xiaweiyicaoqunwu/CFKG_Code.git
cd CFKG_Code
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install "numpy<2" pandas openpyxl "neo4j>=5.26,<6" "py2neo==2021.2.4" "scikit-learn>=1.3" python-dotenv tqdm
python -m pip install "torch>=2.0,<2.2"
python -m pip install "sentence-transformers==2.6.1" "transformers==4.38.2" "huggingface-hub>=0.20,<1" "openai>=1.12,<2"
```

The existing `requirements.txt` specifies `py2neo==2024.1.0`, which is unavailable on [PyPI](https://pypi.org/project/py2neo/); the commands above use `2021.2.4` instead.

Graph import and extraction require Neo4j; the manuscript uses **Neo4j Community 5.26.1**. LLM alignment requires a DashScope API key. Entity merging additionally requires APOC's `mergeNodes` and `mergeRelationships` procedures. PyTorch uses CUDA when available, otherwise CPU.

## Usage

Start each workflow below from the repository root.

### 1. Source data processing

Prepare cleaned XLSX tables, then edit `INPUT_FILES`, `SOURCE`, `ENTITY_RULES`, and `RELATION_RULES` in `raw_data_processing/xlsx2json.py` to match their columns. The script produces entity and relationship JSON files under `json_output/<SOURCE>/`.

```bash
cd raw_data_processing
python xlsx2json.py
python json2neo4j.py --input json_output/Database --check
python json2neo4j.py --input json_output/Database
```

Replace `Database` with your configured `SOURCE`. Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in the terminal before importing; the importer prompts for a password if needed. Source collection, table cleaning, and unit normalization must be completed beforehand. The Figshare exports use a different schema and require a separate import or conversion procedure.

### 2. Entity alignment

**Evaluate the supplied results** without a database, API key, or additional packages:

```bash
cd entity_alignment
python evaluate_alignment.py --test data/evalution/eval_2000.jsonl --undirected
python evaluate_alignment_by_type.py --test data/evalution/eval_2000.jsonl --undirected
```

The test set contains 2,000 pairs, evenly split between positive and negative examples. The supplied predictions give:

| Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0.9055 | 0.8592 | 0.9700 | 0.9112 |

`--undirected` allows either pair direction to match. The directory spelling `evalution` follows the repository.

**Run alignment on your own graph:** create a local `.env` file in `entity_alignment/`:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
```

Check the model settings in `config.py`, then run from `entity_alignment/`:

```bash
python data_extraction.py
python embedding_generation.py
python candidate_generation.py
python llm_alignment.py
```

The defaults use `BAAI/bge-small-en-v1.5` for name embeddings and `qwen3.8-flash` through DashScope for alignment. Candidates are selected within the same entity type using name similarity, with a combined score of `0.8 × name similarity + 0.2 × neighbor Jaccard similarity`, a threshold of 0.7, and up to ten candidates per entity.

To merge aligned entities, back up the database and run from the same directory:

```bash
python data/merge_entities.py
```

Merging updates Neo4j and resolves conflicting properties through the LLM. A cluster remains unmerged if a conflict cannot be resolved. Use the same database snapshot throughout: alignment files contain Neo4j internal IDs, which can change after reimport. Preserve supplied files before regenerating outputs, and keep credentials out of version control.

### 3. Link prediction

Supported models: **TransE, DistMult, ComplEx, RotatE, and ConvE**.

Evaluate a supplied checkpoint without connecting to Neo4j:

```bash
cd link_prediction
python evaluate_baseline.py --model ComplEx
```

To train a model using the supplied split, run from `link_prediction/`:

```bash
python train_baseline.py --model ComplEx
python evaluate_baseline.py --model ComplEx
```

Replace `ComplEx` with another supported model name. The dataset contains 49,946 training, 6,243 validation, and 6,244 test triples. Defaults are a base embedding dimension of 128, 200 epochs, batch size 1,024, Adam with learning rate 0.005, and 256 negative tails per positive triple. Evaluation reports filtered MRR and Hits@1/3/10.

To create a new dataset from Neo4j, set the connection environment variables and run `python data_loader.py` from `link_prediction/`. This script does not load `entity_alignment/.env`. It splits triples 80%/10%/10% with seed 42 and overwrites `data/dataset.pkl`. Keep supplied weights paired with their original dataset; retrain after replacing the dataset. Training also overwrites the corresponding model checkpoint.

## Reproducibility notes

- The current candidate generator takes the top candidates by name similarity before applying the combined score; the manuscript describes ranking by the combined score first.
- LLM calls explicitly set `temperature=0.1`, but do not explicitly set the manuscript's `top_p`, `top_k`, or LLM seed.
- The current link prediction evaluator predicts tails on original directed triples. Reproducing the manuscript protocol requires adding head-prediction queries and averaging three independent training runs; training seeds are not fixed by the scripts.
- Original source XLSX files and generated name embeddings are not included. The raw JSON converter/importer omits the final graph's `subordination` relation, and final export preparation is not fully scripted.
- `data/candidates.jsonl` under `entity_alignment/` is approximately 107 MiB and requires Git LFS or separate distribution under [GitHub's file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## License

Permission is granted to copy and redistribute the code in this repository. Datasets and third-party dependencies remain subject to their respective terms.
