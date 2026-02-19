# Multimodal RAG System for Document Intelligence

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)

A production-grade Retrieval-Augmented Generation (RAG) system for intelligent document processing. This system extracts information from PDF documents using advanced machine learning techniques, performs hybrid retrieval, and generates accurate answers using large language models.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Supported Models](#supported-models)
- [Testing](#testing)
- [Deployment](#deployment)
- [Architecture](#architecture)
- [License](#license)

## Features

### Document Processing
- PDF text extraction with PyMuPDF
- Image extraction and captioning (BLIP)
- Table extraction with structure preservation
- OCR support for scanned documents (Tesseract)

### Retrieval System
- Dense vector search using FAISS
- Sparse retrieval with BM25
- Reciprocal Rank Fusion (RRF) for hybrid results
- Cross-encoder reranking (MS-MARCO)
- Query expansion with synonyms and HyDE

### Production Capabilities
- JWT-based authentication
- API key validation
- Redis caching layer
- Rate limiting per user
- Real-time response streaming
- RAGAS-style evaluation metrics

### Interfaces
- Gradio web interface
- FastAPI REST endpoints
- Command-line interface

## Project Structure

```
multimodal-rag-system/
├── src/
│   ├── api/                # REST API server
│   │   ├── server.py       # FastAPI application
│   │   ├── auth.py         # Authentication module
│   │   └── cache.py        # Caching layer
│   ├── web/                # Web interface
│   │   └── app.py          # Gradio application
│   ├── preprocessing/      # Document processing
│   │   ├── pdf_parser.py   # PDF extraction
│   │   └── chunking.py     # Text chunking
│   ├── embeddings/         # Embedding models
│   ├── retrieval/          # RAG components
│   │   ├── rag_pipeline.py # Main pipeline
│   │   ├── vector_db.py    # Vector store
│   │   └── reranker.py     # Cross-encoder
│   ├── evaluation/         # Metrics
│   └── utils/              # Utilities
├── tests/                  # Unit tests
├── data/                   # Documents
├── artifacts/              # Saved indexes
└── k8s/                    # Kubernetes manifests
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended: 6GB+ VRAM)
- Ollama (for LLM inference)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd multimodal-rag-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download Ollama model
ollama pull llama3.1:8b
```

## Usage

### Command Line Interface

```bash
# Ingest documents
python -m src.main ingest -i data/raw/ -o artifacts/index

# Interactive query mode
python -m src.main query -i artifacts/index --interactive --model llama3
```

### Web Interface

```bash
python -m src.web.app
```

Access the interface at `http://localhost:7860`

### REST API

```bash
python -m src.api.server
```

API documentation available at `http://localhost:8000/docs`

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/ingest/upload` | POST | Upload documents |
| `/ingest/load` | POST | Load existing index |
| `/query` | POST | Submit query |

## Supported Models

| Model | VRAM | Inference Speed | Response Quality |
|-------|------|-----------------|------------------|
| Llama 3.1 8B | 4.9 GB | High | Excellent |
| Mistral 7B | 4.1 GB | High | Excellent |
| Phi-3 Mini | 2.2 GB | Very High | Good |
| Qwen2 1.5B | 0.9 GB | Very High | Moderate |

## Testing

```bash
pytest tests/ -v
```

All 15 tests should pass.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions covering:

- Docker containerization
- Hugging Face Spaces
- AWS EC2 and Google Cloud
- Kubernetes orchestration
- CI/CD pipelines

## Architecture

```
Input Query
    │
    ▼
Query Expansion (Synonyms, HyDE)
    │
    ▼
Hybrid Retrieval (Dense + BM25)
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
LLM Generation (Ollama)
    │
    ▼
Response with Citations
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Average Response Time | ~40 seconds |
| Retrieval Method | Hybrid RRF |
| Reranking Improvement | +15% precision |
| GPU Memory Usage | ~5 GB |

## License

This project is licensed under the MIT License.
