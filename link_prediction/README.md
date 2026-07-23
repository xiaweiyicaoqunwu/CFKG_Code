# Link Prediction

Use this directory to export triples from Neo4j, build the serialized dataset, train a knowledge-graph embedding model, and evaluate its filtered tail-prediction rankings.

Run the commands from `link_prediction/`. `train.py` and `evaluate.py` use `./data` relative to the current working directory.

## Prerequisites

- Python 3.10 or 3.11
- A running Neo4j database when rebuilding `data/dataset.pkl`
- Sufficient memory for the exported triples and evaluation score matrix
- Optional: a CUDA-enabled PyTorch installation

## Installation

PowerShell:

```powershell
cd link_prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Bash:

```bash
cd link_prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the Dataset

Skip this section if the supplied `data/dataset.pkl` is the dataset you want to use.

Set the Neo4j connection variables:

PowerShell:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "<neo4j-password>"
```

Bash:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<neo4j-password>"
```

Export and encode the graph:

```bash
python data_loader.py
```

The command writes `data/dataset.pkl`. It contains:

- `train`, `valid`, and `test` triple arrays
- `entity2id` and `relation2id`
- `id2name`, `id2type`, and `id2attributes`
- `num_nodes` and `num_rels`

The loader creates inverse relations and splits the source triples with a fixed random seed.

## Train a Model

Supported values for `--model`:

- `TransE`
- `DistMult`
- `ComplEx`
- `RotatE`
- `ConvE`

Example:

```bash
python train.py --model ComplEx
```

The trained state dictionary is written to:

```text
data/ComplEx_weights.pth
```

Other model names follow the same pattern: `data/<MODEL>_weights.pth`.

Training uses CUDA automatically when available. The main settings—including embedding size, batch size, epochs, learning rate, negative samples, and adversarial temperature—are constants near the top of `train.py`.

## Evaluate a Model

The corresponding weight file must already exist in `data/`.

```bash
python evaluate.py --model ComplEx
```

The command prints filtered:

- Mean Reciprocal Rank (MRR)
- Hits@1
- Hits@3
- Hits@10

Evaluation scores each test pair against all candidate tail entities. Large datasets can require substantial CPU/GPU memory.

## Use the Model in the Web Service

The Web service expects the dataset and ComplEx weights under different filenames:

```text
flask_citrus/model/dataset.pkl
flask_citrus/model/model_weights.pth
```

From the repository root, after training ComplEx:

PowerShell:

```powershell
Copy-Item link_prediction\data\dataset.pkl flask_citrus\model\dataset.pkl
Copy-Item link_prediction\data\ComplEx_weights.pth flask_citrus\model\model_weights.pth
```

Bash:

```bash
cp link_prediction/data/dataset.pkl flask_citrus/model/dataset.pkl
cp link_prediction/data/ComplEx_weights.pth flask_citrus/model/model_weights.pth
```

Always deploy a dataset and weight file created from the same entity/relation mappings.

## Common Problems

- **`dataset.pkl` is not found:** run from `link_prediction/` and verify `data/dataset.pkl`.
- **A weight file is not found:** train the same model name before evaluating it.
- **State-dictionary size mismatch:** regenerate the weights for the current dataset and use the configured embedding dimension.
- **Evaluation runs out of memory:** reduce `BATCH_SIZE` in `evaluate.py` or run on a machine with more memory.
- **Neo4j export fails:** verify the Bolt URI, credentials, graph size, and available memory.

