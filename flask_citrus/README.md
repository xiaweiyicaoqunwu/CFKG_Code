# CFKG Web Service

This directory contains the Flask application, templates, static assets, Neo4j queries, and the ComplEx model used by the local CFKG Web interface.

Run the application from `flask_citrus/`. The visit counter uses `visits.txt` relative to the current working directory.

## Prerequisites

- Python 3.10 or 3.11
- A running Neo4j database for graph search, entity details, suggestions, and KG-assisted question answering
- A DashScope API key for natural-language-to-Cypher generation and answer generation
- The matching ComplEx dataset and weight files described below

For safer operation, connect the application to a read-only Neo4j account. The application validates generated Cypher, but application-level validation does not replace database permissions.

## Installation

PowerShell:

```powershell
cd flask_citrus
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Bash:

```bash
cd flask_citrus
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required Model Files

The application loads:

```text
model/
├── dataset.pkl
└── model_weights.pth
```

`dataset.pkl` must contain the entity/relation mappings and metadata created by `link_prediction/data_loader.py`. `model_weights.pth` must be a ComplEx state dictionary trained from that same dataset.

If the files are missing or incompatible, the application can start, but `/infer_flavor` is unavailable.

To deploy a newly trained model, see [the link-prediction guide](../link_prediction/README.md#use-the-model-in-the-web-service).

## Configuration

PowerShell:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "<neo4j-password>"
$env:DASHSCOPE_API_KEY = "<dashscope-api-key>"
$env:FLASK_SECRET_KEY = "<random-long-secret>"
```

Bash:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<neo4j-password>"
export DASHSCOPE_API_KEY="<dashscope-api-key>"
export FLASK_SECRET_KEY="<random-long-secret>"
```

Variable behavior:

- `NEO4J_URI`: defaults to `bolt://localhost:7687`.
- `NEO4J_USER`: defaults to `neo4j`.
- `NEO4J_PASSWORD`: required for Neo4j-backed features.
- `DASHSCOPE_API_KEY`: required for Cypher and answer generation.
- `FLASK_SECRET_KEY`: used to sign Flask sessions. If omitted, a new random value is generated at startup, invalidating existing sessions after a restart.

## Start the Local Service

From `flask_citrus/`:

```bash
python app.py
```

Open:

- Home: `http://127.0.0.1:5000/`
- Graph interface: `http://127.0.0.1:5000/graph`
- About page: `http://127.0.0.1:5000/about`
- Download page: `http://127.0.0.1:5000/download`

The download endpoint expects `static/data/graph.json`.

## Feature Availability

| Feature | Neo4j | DashScope key | Model files |
|---|---:|---:|---:|
| Static home/about pages | No | No | No |
| Graph search and details | Yes | No | No |
| Suggestions | Yes | No | No |
| Flavor association prediction | Optional for existing-link classification | No | Yes |
| KG-assisted question answering | Yes | Yes | No |

If a dependency is unavailable, the server logs the condition and the affected endpoint may return an error or reduced result.

## Development and Deployment Notes

- `app.py` starts Flask with `debug=True`; use this only for local development.
- For public deployment, use a production WSGI server, disable debug mode, configure HTTPS, and place secrets in the deployment environment.
- Restrict network access to Neo4j and use a least-privilege, preferably read-only account.
- Do not expose `/test_connection` publicly without an access-control review.
- The local visit counter is file-based and is not suitable for multiple workers.

## Common Problems

- **Neo4j features are disabled:** set `NEO4J_PASSWORD` and verify the database is reachable.
- **Question answering is disabled:** set `DASHSCOPE_API_KEY` in the shell that starts Flask.
- **The model does not load:** verify both model files, their entity/relation counts, and the ComplEx embedding width.
- **The download fails:** verify `static/data/graph.json` exists.
- **Imports fail:** start the application from `flask_citrus/` after activating its virtual environment.

