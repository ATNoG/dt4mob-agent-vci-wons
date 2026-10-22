# DT4MOB Agentic Traffic Monitoring (WONS 2027 Artifact)

Companion code, Goose recipes, deterministic reference pipeline, and evaluation outputs for the paper:

> **Tool-Enforced Determinism for Agentic Traffic Monitoring in a Distributed Mobility Digital Twin**

Part of the **DT4MOB (Digital Twins for Mobility)** project (`2025.00153.DT4ST`), integrated with the [DT4MOB platform](https://github.com/ATNoG/dt4mob-platform).

## Overview

This repository implements an agentic interpretation layer over the DT4MOB highway digital twin (Eclipse Hono, Apache Kafka, Eclipse Ditto) for Porto's VCI corridor (A20/A28), alongside an equivalent deterministic reference pipeline (`classic_runner.py`) that shares the exact same Model Context Protocol (MCP) tool functions (`mcp_server.py`).

- **Agent Orchestration**: Built on [Goose](https://goose-docs.ai/) (Rust CLI agent framework) with declarative YAML recipes (`coordinator_recipe.yaml`, `predictor_subagent_recipe.yaml`, `evaluator_subagent_recipe.yaml`).
- **Tool-Enforced Determinism**: A 12-tool [FastMCP](https://gofastmcp.com/) server (`mcp_server.py`) performs all clock, topology, context retrieval, XGBoost inference, sMAPE deviation calculation, and twin persistence operations.
- **Models Evaluated**: Unsloth 4-bit quantized open-weight LLMs (`gemma-4-26b-a4b` and `qwen3.8-27b`) replayed over 169 consecutive hourly sessions (23–29 September 2024).

## Repository Layout

```text
├── mcp_server.py                  # FastMCP server exposing 12 deterministic tools
├── coordinator_recipe.yaml        # Goose recipe for corridor-level coordination
├── predictor_subagent_recipe.yaml # Goose subagent recipe for per-node forecasting
├── evaluator_subagent_recipe.yaml # Goose subagent recipe for per-node deviation evaluation
├── classic_runner.py              # Deterministic reference pipeline calling the same MCP tools
├── llm_proxy.py                   # OpenAI-compatible proxy recording tokens, caching, latency
├── compare.py                     # Per-session & per-node comparison against classic reference
├── metrics.py                     # Summary metrics exporter
├── run_models.sh                  # Multi-model replay runner (smoke check + 169-session replay)
├── analysis.ipynb                 # Full evaluation notebook producing CSV tables in analysis_out/
├── analysis_out/                  # Aggregated CSV tables (fidelity, agreement, failures, time, cost)
├── scripts/
│   ├── export_numbers.py          # Generates LaTeX macros (numbers.tex) from analysis_out/
│   └── make_figures.py            # Generates paper plots from analysis_out/
├── config.yaml                    # Goose configuration (stdio MCP extension, developer disabled)
├── custom_skynetollama.json       # Custom OpenAI-compatible provider definition
├── Dockerfile                     # Container build (Python 3.12 + Goose CLI v1.29.1)
├── docker-compose.yaml            # Docker Compose stack (Coordinator + MLflow)
├── CITATION.cff                   # Citation metadata
└── LICENSE                        # MIT License
```

## Reproducing the Paper Numbers and Figures

All tables and macros in the paper are generated directly from the CSVs in `analysis_out/`:

```bash
uv run scripts/export_numbers.py --data analysis_out --out numbers.tex
uv run scripts/make_figures.py
```

## Running the Replay

1. Copy and configure `.env` with your DT4MOB Eclipse Ditto, S3/MinIO, and LLM endpoint credentials.
2. Run the replay across both models and the deterministic reference:

```bash
./run_models.sh gemma-4-26b-a4b qwen3.8-27b
```

## License

Released under the [MIT License](LICENSE).
