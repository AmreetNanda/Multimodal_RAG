<img width="358" height="29" alt="image" src="https://github.com/user-attachments/assets/8d9a586b-32f1-40c6-9598-6d3c8d250e1b" /># 🧠 Local Multimodal RAG System (PDF + Images)

**A local, GPU-friendly Multimodal Retrieval-Augmented Generation (RAG) system that ingests PDFs and images, builds hybrid search indices (BM25 + Vector), reranks results using a multimodal LLM (LLaVA via Ollama), and generates concise answers with sources. All with a Streamlit dashboard for metrics and latency visualization.**
---

## 🎯 Goal
Build and deploy a fully local, multimodal Retrieval-Augmented Generation (RAG) system capable of reasoning over PDFs and images. The project focuses on experimenting with modern open-source LLMs while staying within realistic consumer hardware constraints. Emphasis is placed on hybrid retrieval, multimodal reranking, grounded answer generation, and end-to-end system performance evaluation.

## ✨ Features

- 📄 PDF ingestion (text + tables)
- 🖼 Image ingestion (objects, captions, OCR via LLaVA)
- 🔍 Hybrid retrieval 
  - BM25 (keyword)
  - Vector search (semantic embeddings)
- 🧮 Multimodal reranking (text + image context)
- 🧠 LLM answer generation with source references
- 📊 Streamlit dashboard
   - Latency tracking
   - Precision@K / Recall@K
- 💻 CLI-based ingestion & querying
- Fully dockerized multimodal RAG
- 🐳docker-compose
---

## 🧰 Technologies Used:
- Python 3.10+
- Ollama + LLaVA (llava-phi3)
- LangChain (langchain-ollama)
- FAISS (vector search)
- Whoosh (BM25)
- Sentence-Transformers
- PDFPlumber
- Streamlit
- TQDM
---

## Project Structure
```bash
Multimodal_RAG/
├── crawler/                         # Crawl PDFs and Images
│   ├── pdf_crawler.py               # scan local PDFs
│   ├── image_crawler.py             # scan local image files
│   └── __init__.py
│
├── extractor/                       # Extract content
│   ├── pdf_text_extractor.py        # text + tables
│   ├── image_object_extractor.py    # objects, captions, text via LLaVA/BakLLaVA
│   └── __init__.py
│
├── indexer/                         # Build indices
│   ├── bm25_index.py
│   ├── vector_index.py
│   ├── hybrid_index.py
│   └── __init__.py
│
├── reranker/                        # Rerank top-K results
│   ├── multimodal_reranker.py       # combined text + image relevance
│   └── __init__.py
│
├── generator/                       # Generate final answer
│   ├── answer_generator.py          # produce concise answer with references  
│   └── __init__.py
│
├── dashboard/                       # for latency, Precision@K, Recall@K, and diagram previews
│   ├── streamlit_app.py
│   ├── latency_tracker.py
│   ├── ranking_metrics.py
│   └── __init__.py
│
├── scripts/
│   ├── ingest.py                    # crawl + extract + build indices
│   ├── run_query.py                 # interactive multimodal Q&A 
│   └── __init__.py
│
├── data/
│   ├── pdfs/
│   └── images/
│
├── index/
│   ├── bm25/
│   ├── vector/
│   └── hybrid/
│
├── utils/
│   ├── logger.py
│   ├── helpers.py
│   ├── text_clean.py
│   └── __init__.py
│
├── requirements.txt
└── README.md

```

## 🏗 Architecture Overview
```bash
   User query
   ↓
   Hybrid Search (Bm25 + Vector)
   ↓
   Multimodal Reranker (LLaVA)
   ↓
   Answer Generator (LLaVA)
   ↓
   Final Answer + Sources
```
## 🛠 Installation

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Multimodal_RAG.git
cd Multimodal_RAG
```
### 2. Install dependencies
```bash
pandas
numpy
scikit-learn
langchain
langchain_community
langchain-core
langchain-classic
langchain_ollama
langchainhub
streamlit
sse_starlette
faiss-cpu
pdfplumber
sentence-transformers
requests
tqdm
whoosh

pip install -r requirements.txt
```

### 3. Build plan 
#### **Step 1: Ingest data**
```bash
python -m scripts.ingest --pdf_dirs data/pdfs --image_dirs data/images

Expected output:-
Found PDF: ...
Extracted text from ...
Found image: ...
Extracted image content ...
Hybrid index saved to index/hybrid/hybrid_index.pkl

```

#### **Step 2: Query pipeline**
```bash
python -m scripts.run_query

Example:

Enter your query:
What does the system architecture diagram describe?

```
#### **Step 3: Launch dashboard**
```bash
streamlit run dashboard/streamlit_app.py

Open in your browser:
👉 http://localhost:8501/
👉 Precision and recall results with bar chart for latency
```

## 🐳 Running with Docker (optional)
### Build the docker image
```bash
docker build -t multimodal-rag .
```

### Run the container
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/index:/app/index \
  multimodal-rag
```
Open: 👉 http://localhost:8501

### Run with GPU (optional)
```bash
docker run --gpus all \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/index:/app/index \
  multimodal-rag
```

### Ingest Data Inside Docker
```bash
docker exec -it <container_id> \
  python -m scripts.ingest \
  --pdf_dirs data/pdfs \
  --image_dirs data/images
```

### Run Query CLI Inside Docker
```bash
docker exec -it <container_id> \
  python -m scripts.run_query
```


### 🐳 Docker Compose Section
#### **Step 1: Start Everything**
```bash
docker compose up --build
```
This starts:
- Ollama at http://localhost:11434
- Streamlit UI at http://localhost:8501

#### **Step 2: Pull LLaVA Model (First Time Only)**
In another terminal:
```bash
docker exec -it ollama ollama pull llava-phi3
docker exec -it ollama ollama pull bakllava (optional)
```

#### **Step 3: Ingest Documents**
```bash
docker exec -it multimodal-rag \
  python -m scripts.ingest \
  --pdf_dirs data/pdfs \
  --image_dirs data/images
```

#### **Step 4: Run Query CLI**
```bash
docker exec -it multimodal-rag \
  python -m scripts.run_query
```

#### **Step 5: Stop Services**
```bash
docker compose down
```

## Screenshots
![App Screenshot](https://github.com/AmreetNanda/Multimodal_RAG/blob/main/Screenshot-1.png)
![App Screenshot](https://github.com/AmreetNanda/Multimodal_RAG/blob/main/Screenshot-2.png)

## Demo 
https://github.com/user-attachments/assets/f348c8c5-53df-4b2b-a33c-73bc2a35906f

## License
[MIT](https://choosealicense.com/licenses/mit/)
## Demo

