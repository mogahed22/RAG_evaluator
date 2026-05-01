# Mini RAG Analytics Project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) prototype for question answering and retrieval evaluation using a local vector database. It is built around:
- `Chroma` for document embedding storage and similarity search
- `Ollama` for local embedding and language model inference
- `LangChain` components for text splitting, retrieval, and prompt composition

The repository includes data preparation, retrieval evaluation, generation evaluation, and performance sweep analysis.

## Repository Structure

- `populate_database.py` — prepares and indexes the dataset into a Chroma vector store
- `query_data.py` — demonstrates RAG-style query execution and response generation
- `eval_retrieval.py` — evaluates retrieval quality using Hit@K and MRR metrics
- `eval_generation.py` — evaluates generation correctness against expected answers
- `similarity_pipeline.py` — helper functions for similarity search and context assembly
- `get_embedding_function.py` — defines the Ollama embedding function for the vector database
- `sweep_analyzer.py` — performs chunk-size sweeps and visualizes retrieval metric impact
- `data/golden_test_set_squad.csv` — evaluation dataset used by the project
- `chroma/` — persisted local Chroma database files
- `results/` — generated evaluation outputs and visualizations
  - `results/sweep_comparison_results.csv` — chunk-size sweep metrics comparison
  - `results/sweep_line_chart.png` — performance chart showing retrieval metrics

## Prerequisites

- Python 3.10+ recommended
- Local Ollama installation with model availability for:
  - `mistral` (LLM)
  - `nomic-embed-text` (embedding model)
- `chroma` directory will be created automatically for persistence

## Setup

1. Clone or copy the repository.
2. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies. Example:

```bash
pip install pandas seaborn matplotlib langchain langchain-chroma langchain-ollama langchain-text-splitters langchain-community
```

4. Verify that Ollama is running and the required models are accessible.

## Usage

### 1. Populate the Chroma database

Build the vector store from `data/golden_test_set_squad.csv`:

```bash
python3 populate_database.py --chunk_size 100 --chunk_overlap 20
```

Customize chunking parameters with `--chunk_size` and `--chunk_overlap`.

### 2. Run a RAG query

Use `query_data.py` to run a sample question through retrieval and generation:

```bash
python3 query_data.py
```

### 3. Evaluate retrieval performance

Run retrieval evaluation with a random sample of questions:

```bash
python3 eval_retrieval.py
```

This script computes Hit@1, Hit@3, Hit@5, and Mean Reciprocal Rank (MRR).

### 4. Evaluate generation performance

Run generation evaluation to compare model answers against expected answers:

```bash
python3 eval_generation.py
```

Results are written to `results/generation_status.txt`.

### 5. Sweep chunk-size performance

Use `sweep_analyzer.py` to compare retrieval results across chunk sizes and save a visualization:

```bash
python3 sweep_analyzer.py
```

The sweep output is saved to:
- `results/sweep_comparison_results.csv` — detailed sweep metrics for each chunk size
- `results/sweep_line_chart.png` — line chart visualizing Hit@K and MRR results

## Notes

- `eval_retrieval.py` uses paraphrasing to create query variants before similarity search.
- `query_data.py` can be extended to support richer prompt templates, additional query generation, or multiple retrieval strategies.
- `populate_database.py` currently loads data from CSV and creates Chroma document chunks for retrieval.

## Recommendations

- Keep the `chroma` directory in source control only if you want to preserve the indexed database state.
- If using a different LLM or embedding provider, update `get_embedding_function.py` and `query_data.py` accordingly.
- For production use, add explicit dependency management (e.g., `requirements.txt`) and environment setup automation.

## Contact

For improvements or issues, update the repository with:
- more robust evaluation logging
- alternative retrieval scoring methods
- extended RAG prompt engineering
