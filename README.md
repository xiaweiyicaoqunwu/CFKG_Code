# CFKG Code

Code accompanying *A multi-source knowledge graph of citrus sensory properties*.

This repository contains three independently installable modules. Use the module-level guides for complete commands and configuration details.

## Repository Layout

```text
cfkg_code/
├── entity_alignment/   # Extract, embed, rank, and align Neo4j entities
├── link_prediction/    # Prepare triples, train models, and evaluate them
└── flask_citrus/       # Run the local CFKG Web service
```

## Recommended Workflow

1. Install or restore a Neo4j database containing the CFKG data.
2. Run the entity-alignment pipeline only when rebuilding or aligning graph entities.
3. Export a link-prediction dataset from Neo4j, train a model, and evaluate it.
4. Put the selected ComplEx dataset and weights in `flask_citrus/model/` using the filenames expected by the Web service.
5. Configure the required environment variables and start the Flask application.

The repository already includes `link_prediction/data/dataset.pkl` and the two model files used by the Web service. Rebuild these artifacts only when the graph or model needs to be updated.

## Shared Prerequisites

- Python 3.10 or 3.11 is recommended.
- Neo4j must be reachable over Bolt.
- A DashScope API key is required only for LLM-assisted entity alignment and Web question answering.
- A CUDA-enabled PyTorch installation is optional. The code falls back to CPU where supported.

Create a separate virtual environment for each module because their dependencies and workloads differ.

## Shared Environment Variables

The modules read credentials from environment variables. Do not write real credentials into source files.

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

`DASHSCOPE_API_KEY` is not needed for link-prediction training or evaluation.

## Module Documentation

- [Entity alignment](entity_alignment/README.md) — [dependencies](entity_alignment/requirements.txt)
- [Link prediction](link_prediction/README.md) — [dependencies](link_prediction/requirements.txt)
- [CFKG Web service](flask_citrus/README.md) — [dependencies](flask_citrus/requirements.txt)

## Moving a Trained Model to the Web Service

After training ComplEx, copy the dataset and weights to the names loaded by `flask_citrus/app.py`.

PowerShell, from the repository root:

```powershell
Copy-Item link_prediction\data\dataset.pkl flask_citrus\model\dataset.pkl
Copy-Item link_prediction\data\ComplEx_weights.pth flask_citrus\model\model_weights.pth
```

Bash, from the repository root:

```bash
cp link_prediction/data/dataset.pkl flask_citrus/model/dataset.pkl
cp link_prediction/data/ComplEx_weights.pth flask_citrus/model/model_weights.pth
```

The dataset and weight file must come from the same training run or schema. A mismatch in entity count, relation count, or embedding width prevents the Web service from loading the model.

