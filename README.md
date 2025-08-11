# Business Docs AI Assistant 🤖📄

This is a custom ChatGPT-style assistant designed to answer questions about internal business documents.

This project uses local embeddings (all-MiniLM-L6-v2) to simulate a full RAG pipeline with FAISS, enabling document-based QA without API costs. It’s extensible to OpenAI embeddings, but I intentionally kept it local-first to show practical offline AI engineering.

## Features
- Loads and processes PDFs
- Splits documents into chunks for context-aware Q&A
- Uses OpenAI GPT / Hugging Face models (2 adaptations depending on the budget) for answering questions
- Terminal-based chatbot interface

## Project description
http://bit.ly/46Rj50C

## Installation
```bash
pip install -r requirements.txt
