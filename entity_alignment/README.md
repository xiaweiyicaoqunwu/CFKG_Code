# Entity Alignment

Run the scripts in this directory in the order shown below. The scripts use paths relative to the current working directory, so start by changing into `entity_alignment/`.

## Prerequisites

- Python 3.10 or 3.11
- A running Neo4j database containing the nodes to align
- A DashScope API key for the LLM alignment stage
- Internet access on the first embedding run to download `BAAI/bge-small-en-v1.5`
- Optional: the Neo4j APOC plugin for the entity-merge script

## Installation

PowerShell:

```powershell
cd entity_alignment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Bash:

```bash
cd entity_alignment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Set the connection and API variables before running the pipeline.

PowerShell:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "<neo4j-password>"
$env:DASHSCOPE_API_KEY = "<dashscope-api-key>"
```

Bash:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<neo4j-password>"
export DASHSCOPE_API_KEY="<dashscope-api-key>"
```

`config.py` contains the file paths and processing parameters:

- `BATCH_SIZE`: Neo4j extraction and similarity-computation batch size
- `EMBEDDING_MODEL`: Sentence Transformers model name
- `SIMILARITY_THRESHOLD`: minimum combined candidate score
- `TOP_K`: maximum semantic candidates considered per entity
- `NAME_WEIGHT` and `STRUCTURE_WEIGHT`: candidate-score weights

## Run the Pipeline

All commands below assume that the current directory is `entity_alignment/`.

### 1. Extract nodes and one-hop neighbors

```bash
python data_extraction.py
```

Output: `data/nodes.jsonl`

Each line contains a Neo4j node ID, name, first label, and named one-hop neighbors.

### 2. Generate name embeddings

```bash
python embedding_generation.py
```

Input: `data/nodes.jsonl`  
Output: `data/embeddings.pt`

The script uses CUDA when available and otherwise uses CPU.

### 3. Generate alignment candidates

```bash
python candidate_generation.py
```

Inputs: `data/nodes.jsonl`, `data/embeddings.pt`  
Output: `data/candidates.jsonl`

Candidates are compared only with entities that have the same Neo4j label.

### 4. Run LLM alignment

```bash
python llm_alignment.py
```

Input: `data/candidates.jsonl`  
Output: `data/alignment_results.jsonl`

The output file is opened in append mode. IDs already saved in that file are skipped on later runs. An entity for which no alignment was saved may be processed again.

## Optional: Merge Aligned Neo4j Entities

`data/merge_entities.py` changes the Neo4j database and uses APOC node-merging procedures. It is not required to generate alignment results.

Before running it:

1. Back up the Neo4j database.
2. Review the script and its database configuration.
3. Ensure that APOC is installed and that `apoc.refactor.mergeNodes` is permitted.
4. Verify `data/nodes.jsonl` and `data/alignment_results.jsonl`.
5. Use a database account that is authorized to update and merge nodes.

Run from `entity_alignment/`:

```bash
python data/merge_entities.py
```

This operation is destructive and cannot be treated as a read-only validation step.

## Generated Files

```text
data/
├── nodes.jsonl
├── embeddings.pt
├── candidates.jsonl
└── alignment_results.jsonl
```

These files can be large and may contain graph data. Review them before committing or distributing them.

## Common Problems

- **Neo4j connection fails:** verify the Bolt URI, credentials, database status, and network access.
- **Embedding model download fails:** verify internet access and Hugging Face availability.
- **CUDA runs out of memory:** reduce `BATCH_SIZE` in `config.py` or run on CPU.
- **LLM stage is disabled:** verify that `DASHSCOPE_API_KEY` is set in the same shell.
- **No candidates are produced:** inspect entity labels, names, the similarity threshold, and the generated embeddings.

