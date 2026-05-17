---
title: SHL Assessment Recommender
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SHL Conversational Assessment Recommender API

This is a production-grade FastAPI application that provides conversational AI recommendations for SHL assessments based on user queries, utilizing hybrid FAISS + BM25 retrieval.

## Design Writeup
**Design Choices:** We chose a stateless FastAPI architecture to ensure scalability. The system extracts conversation state on every turn rather than maintaining memory, avoiding state-sync issues. Deployment utilizes a CPU-only PyTorch Docker image to drastically reduce cloud RAM overhead.

**Retrieval:** We implemented a hybrid search: Dense vectors (all-MiniLM-L6-v2 + FAISS) capture semantic meaning, while Sparse (BM25) ensures precise keyword matching for skills. A dictionary query expander bridges the gap between job titles (e.g., "Java developer") and underlying skills.

**Prompt & Evaluation:** The prompt enforces strict constraints (only SHL assessments, max 10 recommendations, mandatory clarification for vague queries). Hallucinations are actively prevented using a post-generation validation layer that filters out any generated URL not found in the raw `catalog.json`.

**What didn't work:** Our initial deploy on 512MB free-tier containers (Render/Railway) crashed instantly due to PyTorch CUDA weights and bulk FAISS encoding on startup. We solved this by forcing CPU-only wheels (`torch+cpu`), pre-computing indices, and pivoting to Hugging Face Spaces (16GB RAM) with automated CI/CD via GitHub Actions.

**AI Usage:** Antigravity (Agentic AI) was utilized to structure the hybrid retrieval, diagnose complex NumPy/PyTorch dependency chains, and construct the CI/CD pipeline.
