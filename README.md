# Intelligent Complaint Analysis for Financial Services

## Overview


## Key Features
- Exploratory data analysis across product categories and complaint structure
- Customisable text chunking strategy with overlap control.
- SentenceTransformer embeddings using `all-MiniLM-L6-v2`.
- Semantic search over ChromaDB with retrievable metadata. 
- Embedding via sentence-transformers/all-MiniLM-L6-v2.
- RAG pipeline powered by LangChain with refined prompt engineering.
- Interactive Gradio UI with real-time querying and context display.
- Evaluation matrix with quality scores and commentary.
- CI-integrated test suite for reproducibility and robustness.

## Table of Contents
- [Project Background](#project-background)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Project Status](#project-status)

## Project Background

## Data Sources

## Project Structure
The repository is organised as follows:
```
├──.dvc/
├──.github/workflows
├── data/
│   ├── raw/                         # Original CFPB dataset
│   ├── processed/                   # Processed datase
│   |   ├── filtered_complaints.csv      
|   |   └── chunked_complaints.csv
│   └── evaluation/
│       ├── rag_evaluation.csv       # Manual scoring
│       └── rag_evaluation.md        # Markdown table
├── notebooks/                       # Notebooks
|   ├── 01_eda_preprocessing.ipynb
|   ├── 02_1_chunking.ipynb
|   └── ...
├── plots/
├── scripts/
│   ├── _01_preprocess.py            # Data cleaning and filtering
│   ├── _02_1_chunk.py               # Text chunking logic
|   └── ...
├── tests/                           # Pytest test suite
│   ├── _01_preprocess.py            # Data cleaning and filtering
│   ├── _02_1_chunk.py               # Text chunking logic
|   └── ...
├── vector_store/                    # Persisted ChromaDB index (excluded from Git)
├── .dvcignore
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8 or newer (recommended)
- `pip` (Python package manager)
- [DVC](https://dvc.org/) (for data version control)
- [Git](https://git-scm.com/)

### Steps
1. **Clone the repository:**
    ```
    git clone https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services.git

    cd Intelligent-Complaint-Analysis-for-Financial-Services
    ```

2. **Create a virtual environment:**
   ```
    python -m venv .chatvenv

    # On Windows:
    .chatvenv\Scripts\activate

    # On Unix/macOS:
    source .chatvenv/bin/activate
    ```

3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

4. **(Optional) Set up DVC:**
    ```
    dvc pull
    ```

## Usage
1. **Prepare data**
   - Place complaint data in the `data/raw` directory, ensuring it matches the expected format.

2. **Run jupyter notebooks**
   - Run notebook in `notebooks/` in chronological order. 

3. **View curves**
   - Generated plots will be available in the `plots/` directory or as specified by your script parameters.

4. **Scripts and Tests**
   - Explore `scripts/` and `tests/` directory to interact with script and test suits.

5. **Interactive APP** 
   ```
   python app.py
   ```
   - Features:
      - Input box for natural-language questions
      - Answer powered by RAG + semantic retrieval
      - Source chunks displayed beneath answer
      - Optional streaming with token-by-token output
6. **Evaluation Table**
   - Located at `data/evaluation/rag_evaluation.csv`. It includes: 
      - Questions used
      - Answers generated
      - Source documents pulled
      - Manual score (1–5)
      - Commentary

## Contribution

## Project Status
Final submission merged. Checkout the commit history [here](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/commits?author=nuhaminae). 
