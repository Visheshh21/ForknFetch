# Fork & Fetch 🍴
Fork & Fetch is a Retrieval-Augmented Generation (RAG) powered recipe assistant. It enables users to search for recipe suggestions based on their queries using a combination of semantic search and generative AI. Think of it as your smart AI-powered cookbook!

## 📌 Features
- 🔍 Semantic Recipe Search using FAISS and HuggingFace sentence-transformers.
- 🍳 Intelligent Recipe Suggestions powered by LLaMA 3 through Ollama.
- 🧠 Uses LangChain's RetrievalQA chain to combine retrieved recipes with LLM reasoning.
- 🎨 Built with Streamlit for an interactive UI.

## 📂 Project Structure
- [**Dataset_preprocessing.ipynb:**](https://github.com/Kr1mson/ForknFetch/blob/main/Dataset_preprocessing.ipynb) Handles the cleaning and preparation of the recipe dataset for embedding and retrieval.
- [**Embedder.py:**](https://github.com/Kr1mson/ForknFetch/blob/main/embedder.py) Demonstrates the process of creating embeddings from the recipe dataset using HuggingFace's sentence transformers.
- [**Fork&Fetch.py:**](https://github.com/Kr1mson/ForknFetch/blob/main/Fork%26Fetch.py) Contains the complete RAG implementation for recipe search and recommendation with a user-friendly Streamlit UI.

## 📚 Dataset
The recipe dataset was sourced from [**Eight Portions**](https://eightportions.com/datasets/Recipes/#fn:1) and contains approximately 125,000 recipes scraped from various food websites. Each recipe in the dataset includes:
- Recipe title
- List of ingredients and measurements
- Instructions for preparation
- Source URL
- Picture (in some cases)

## 🛠️ Technology Stack
* **Vector Database**: FAISS for efficient similarity search
* **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
* **LLM**: Ollama with llama3.2 model
* **RAG Framework**: LangChain
* **Frontend**: Streamlit
* **GPU Acceleration**: CUDA support (when available)
