# 5-Mini-Projects
A collection of practical AI mini projects showcasing embeddings, semantic search, vector databases, classification, and local LLM applications built with HuggingFace, LangChain, and Python.


🧠 NLP Mini Projects Collection

A collection of hands-on mini projects for practising core NLP techniques such as text classification, sentiment analysis, named entity recognition, semantic embeddings, and vector search. Built with Python, HuggingFace Transformers, SentenceTransformers, PyTorch, and ChromaDB.


⚙️ Setup
git clone https://github.com/K-Ksaveras/5-Mini-Projects.git
cd 5-Mini-Projects

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt


📌 Projects Included

1️⃣ Text Classification (Starter Project):
Zero-shot text classification using facebook/bart-large-mnli.
Introduces transformer pipelines and dynamic label classification.

2️⃣ Sentiment Analysis (Starter Project):
Movie review sentiment analysis using distilbert-base-uncased-finetuned-sst-2-english.
Demonstrates batch inference and confidence scoring.

3️⃣ Named Entity Recognition (Starter Project):
Entity extraction using dslim/bert-base-NER.
Identifies and groups entities such as Person, Organization, and Location.

4️⃣ Semantic Similarity Search:
Text embeddings and semantic search using all-MiniLM-L6-v2 and ChromaDB.
Includes t-SNE visualization and similarity scoring.

5️⃣ CSV Embeddings with Pandas:
Embeddings for structured data and semantic search over CSV files.
Demonstrates vector indexing and retrieval with ChromaDB.



▶️ Run Projects

Python scripts:
python text_classification.py
python sentiment_analysis.py
python named_entity_recognition.py

Jupyter notebooks:
jupyter notebook semantic_similarity_search.ipynb
jupyter notebook csv_embeddings_pandas.ipynb
