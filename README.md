# DocuMiner AI: Legal Document AI Tool

A full-stack AI-powered web application that performs **Retrieval Augmented Generation (RAG)** on legal documents for advanced **summarization** and **question answering**. Built with a focus on scalability, accuracy, and speed, this project leverages modern LLMs, cloud infrastructure, and a clean RESTful API to offer real-time legal document insights.

## Features

* **AI-Powered Analysis**: Provides both **extractive** and **abstractive** answers to legal questions using fine-tuned LLMs.
* **Summarization**: Generates concise, structured summaries of lengthy legal documents.
* **High Accuracy & Speed**: \~85% accuracy with response times under 2 seconds.
* **RAG Pipeline**: Integrates semantic search and generative reasoning for contextual legal queries.

## Tech Stack

### Backend

* **Python**, **Django** (REST Framework)
* **LangChain** for embedding creation and RAG orchestration
* **PGVector** for vector storage
* **OpenAI** & **Gemma (Google)** LLMs (via Hugging Face Inference Endpoints)
* **PostgreSQL** for storing chat and document data
* **Gunicorn** as the WSGI server

### Frontend

* **React** for web support
* **RESTful API** integration for document upload, chat, and querying

### Deployment & DevOps

* **Render** for full-stack deployment (backend, frontend, database)
* **Hugging Face Inference Endpoints** for fast and scalable LLM hosting

## Core Architecture

1. **Document Upload & Preprocessing**
   → User uploads legal documents via UI → stored in PostgreSQL → embeddings created with PGVector.

2. **Query Engine**
   → LangChain-powered semantic search → top-k retrieval → prompt assembly for LLM → answer generation using fine-tuned Gemma model.

3. **Summarization Engine**
   → Documents passed through the fine-tuned gpt-4o-mini LLM model → summary returned in under 2s.

## Performance

| Metric           | Result                           |
| ---------------- | -------------------------------- |
| Summary Accuracy | \~85% (based on benchmarks)      |
| Response Time    | < 2 seconds                      |
| Scalability      | Horizontally scalable via Render |


# Follow the link below to see the entire DocuMiner AI project with set up instructions!
## https://github.com/DocuMinerAI
