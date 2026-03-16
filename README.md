# ScholarAI — RAG-Powered AI Study Assistant

ScholarAI is an end-to-end AI-powered study assistant built on a Retrieval-Augmented Generation (RAG) architecture. It ingests course PDFs into a vector database and enables conversational Q&A grounded in source documents — minimizing LLM hallucinations by constraining responses to retrieved content rather than relying on parametric model memory.

> Built as a personal learning project to deeply understand RAG pipelines, vector databases, LLM orchestration, and modern AI application development.

---

## Motivation

Large Language Models are powerful but prone to hallucinations — generating confident but incorrect answers from their training memory. RAG solves this by retrieving factual context from a trusted knowledge base before generating a response. This project was built to understand RAG architecture end-to-end by implementing it from scratch, explore vector databases and semantic search, and build a practical tool for studying Stanford University course materials covering Machine Learning, Deep Learning, NLP, Transformers, PyTorch, and Neural Networks.

---

## Features

ScholarAI supports conversational Q&A over course PDFs using OpenAI GPT-4o, with responses grounded strictly in retrieved source documents to prevent hallucinations. It maintains conversational memory across multiple questions in a session and supports two retrieval modes — chapter-level and full-course vector search. Each query also auto-suggests relevant YouTube videos to supplement learning. The interface is a clean dark-mode Streamlit web app with course and chapter selectors.

---

## Architecture

The project is split into two main pipelines. The ingestion pipeline parses PDF files using the Unstructured library, splits them into overlapping chunks using LangChain's CharacterTextSplitter, generates 384-dimensional dense vector embeddings using HuggingFace's all-MiniLM-L6-v2 sentence transformer model, and stores them in two persistent ChromaDB vector stores — one combining all course PDFs and one per individual chapter.

The retrieval and generation pipeline takes a user query, embeds it using the same model, performs MMR (Maximal Marginal Relevance) semantic search to retrieve the most relevant and diverse chunks, and passes them along with chat history to OpenAI GPT-4o via LangChain's ConversationalRetrievalChain. ConversationBufferMemory maintains session context so follow-up questions are understood in the right context.

---

## Tech Stack

The project is built entirely in Python and uses LangChain for pipeline orchestration, OpenAI GPT-4o as the large language model accessed via API, ChromaDB as the persistent vector store, HuggingFace Transformers and Sentence Transformers for generating embeddings, Unstructured for PDF parsing and text extraction, PyTorch as the deep learning backend, Streamlit for the web interface, and youtube-search-python for video recommendations.

---

## Project Structure

The repository contains a src folder with five Python files. main.py is the Streamlit chat interface. vectorize_book.py handles the full PDF ingestion and embedding pipeline. vectorize_script.py is the entry point to trigger vectorization. chatbot_utility.py is a helper that reads chapter names from the data directory. get_yt_video.py handles YouTube search and returns video titles and links per query. The data folder holds the course PDFs, and the vector_db and chapters_vector_db folders are auto-generated when the vectorization script runs. A .env file stores API keys and is excluded from GitHub via .gitignore.

---

## Setup & Installation

Clone the repository and create a Python virtual environment. Install PyTorch separately using the official CPU wheel index before running pip install on the requirements file, as PyTorch is not available on the standard PyPI index. Create a .env file in the root directory with your OpenAI API key, the course name matching your data folder, and the device set to cpu. Place your course PDFs inside the data/ml_and_dl folder, then run vectorize_script.py from the src directory to parse, embed, and store all documents. Once vectorization completes, launch the app with streamlit run src/main.py.

---

## How RAG Reduces Hallucinations

Without RAG, a user query goes directly to the LLM which answers from its training weights — it may confidently produce incorrect or outdated information. With RAG, the query is first used to retrieve the most semantically relevant chunks from the vector database. These chunks are injected into the LLM prompt as context, so the model generates an answer grounded in the actual source material rather than its parametric memory. MMR search further improves this by ensuring the retrieved chunks are not redundant, giving the LLM a broader and more useful context window.

---

## Data

Course materials were sourced from Stanford University curriculum covering Machine Learning fundamentals, Deep Learning, Natural Language Processing, the Transformer architecture and Attention Mechanism, PyTorch, and Neural Networks. The PDF files are not included in this repository. You can substitute any course PDFs of your choice by placing them in the data folder and re-running the vectorization script.

---

## Key Learnings

Building this project reinforced how grounding LLM responses in a vector store dramatically reduces hallucinations compared to relying on model memory alone. It also demonstrated how chunking strategy and overlap size directly affect retrieval quality and context continuity across chunks. Handling metadata from the Unstructured library required filtering nested coordinate objects that ChromaDB cannot store. Managing version compatibility across the LangChain ecosystem — particularly between langchain-core, langchain-openai, and langchain-huggingface — required careful pinning to avoid import errors at runtime.

---

## Future Improvements

Planned improvements include support for multiple courses and direct PDF upload through the UI, swapping GPT-4o for open-source LLMs such as Llama or Mistral, adding source citation to show which document and page each answer came from, implementing retrieval evaluation metrics, and containerizing the application with Docker for easier deployment.

---

## Author
Naga Donepudi

> *Built for learning. Powered by curiosity.*
