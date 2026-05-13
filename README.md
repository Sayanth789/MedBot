# 🩺 MedBot – AI Medical QA Assistant

MedBot is an AI-powered medical question-answering system built using **LangChain**, **Llama 3 (8B)**, and a custom medical knowledge base called **MedBook**.

It allows users to ask medical questions and get answers based on trusted contextual data extracted from medical text sources.

---

## 🚀 Features

- 📄 Answers medical questions from a knowledge base (MedBook)
- 🧠 Powered by Llama 3 (8B) language model
- 🔍 Retrieval-Augmented Generation (RAG) pipeline
- 📚 Uses vector search for context-aware responses
- 💬 Simple interactive interface (Streamlit / CLI depending on setup)

---

## 🧠 How it Works


1. Medical book [`MedBook`](data/MedBook.pdf) is loaded and split into chunks  
2. Text is converted into embeddings  
3. Stored in a vector database (e.g., FAISS / Pinecone)  
4. User question is embedded and matched with relevant chunks  
5. Llama 3 generates a final answer using retrieved context  

---

## 🛠️ Tech Stack

- Python 🐍
- LangChain 🦜
- Llama 3 (8B)
- Hugging Face / Ollama (depending on setup)
- FAISS / Pinecone (vector database)
- Flask (UI)

---

## 📦 Project Structure
