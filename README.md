# Intelligent Complaint Analysis for Financial Services

[![CI](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/actions/workflows/CI.yml)
![Version Control](https://img.shields.io/badge/Artifacts-DVC-brightgreen)
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)

## Overview

Intelligent Complaint Analysis for Financial Services is an advanced platform tailored for the automatic analysis, categorisation, and reporting of customer complaints within the financial sector. The system aims to streamline the identification of key issues, sentiment, and trends in customer feedback by leveraging natural language processing (NLP) and machine learning techniques. Ultimately the complaint analysis system helps financial institutions to improve their service quality and regulatory compliance.

---

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

---

## Table of Contents

- [Project Background](#project-background)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Insights](#insights)
- [Contribution](#contribution)
- [Project Status](#project-status)

---

## Project Background

The financial services industry faces a high volume of customer complaints daily, many of which contain crucial insights into systemic issues, regulatory risks, or opportunities for improvement. Manual analysis is labour-intensive and prone to oversight.

This project aims to reduce that time from days to minutes by building a RAG-powered chatbot that retrieves relevant complaint excerpts and generates grounded answers. The system is designed to support internal teams in making faster, evidence-based decisions.

---

## Business Impact

This project transforms that workflow by introducing a Retrieval-Augmented Generation (RAG) chatbot that enables:

### 🔹 Faster Decision-Making

- Internal teams can now query complaints in natural language and receive grounded, formal responses in seconds.
- Multi-turn support allows deeper exploration of issues without restarting the query.

### 🔹 Improved Visibility

- Metadata-enriched embeddings allow filtering by product and issue category, helping teams pinpoint recurring pain points.
- Sentence-aware chunking ensures semantic fidelity, surfacing the most relevant complaint excerpts.

### 🔹 Operational Efficiency

- Manual review time reduced significantly for common queries. The chatbot can handle these in seconds.
- The chatbot supports up to 3 follow-up questions, streamlining multi-layered investigations.

### 🔹 Regulatory Readiness

- Complaint narratives are traceable to their source, supporting auditability and compliance reporting.
- Optional grammar and factuality scoring ensures responses meet internal communication standards.

### 🔹 Scalable Deployment

- The system is modular, CI-integrated, and ready for deployment via Hugging Face Spaces.
- Non-technical users can interact via a Gradio UI with dropdown filters and source transparency.

---

## Data Sources

The project uses real-world complaint data from the Consumer Financial Protection Bureau (CFPB). Each record includes:

- A free-text narrative from the consumer
- Product and company metadata
- Submission dates and issue labels

The dataset was filtered to include only the five target products (Credit card, Personal loan, Buy Now, Pay Later (BNPL), Savings account, Money transfers) and cleaned to remove empty narratives.

---

## Project Structure

The repository is organised as follows:

```bash
├── .chatvenv/                         # Virtual environment (not pushed)
├── .dvc/                              # Data Version Control
├── .github/workflows                  # CI workflows
├── data/                              # Data files
├── notebooks/                         # Notebooks
├── plots/                             # Plots and charts for reporting
├── scripts/                           # Core scripts
├── tests/                             # Unit and pytest
├── .dvcignore                         # Ignore DVC files
├── .flake8                            # Flake8 configuration
├── .gitignore                         # Ignore unnecessary files
├── .pre-commit-config.yaml            # Pre-commit configuration
├── .trunkignore                       # Ignore trunk files
├── .yamllint.yml                      # YAML lint configuration
├── app.py                             # Main Gradio application script
├── format.ps1                         # Formatting script
├── pyproject.toml                     # Project configuration
├── README.md                          # Project overview and setup instructions
└── requirements.txt                   # Python package dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or newer (Python 3.11 recommended)
- `pip` (Python package manager)
- [DVC](https://dvc.org/) (for data version control)
- [Git](https://git-scm.com/)

---

### Setup

```bash
# Clone repo
git clone https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services.git
cd Intelligent-Complaint-Analysis-for-Financial-Services
____________________________________________
# Create and activate virtual environment
python -m venv .chatvenv
.chatvenv\Scripts\activate      # On Windows
source .chatvenv/bin/activate   # On Unix/macOS
____________________________________________
# Install dependencies
pip install -r requirements.txt
____________________________________________
# Install and activate pre-commit hooks
pip install pre-commit
pre-commit install
____________________________________________
# (Optional) Pull DVC data
dvc pull
```

---

## Usage

1. **Prepare data**
   - Place complaint data in the `data/raw` directory, ensuring it matches the expected format.

2. **Run jupyter notebooks**
   - Open with Jupyter or VSCode to navigate the workflow interactively.
   - Run notebook in `notebooks/` in chronological order for exploratory and iterative development:

      - Run the EdA notebook to explore the data and perform initial preprocessing.

      ```bash
      notebooks/01_eda_preprocessing.ipynb
      ```

      - Run the chunking notebook to create text chunks for embedding.

      ```bash
      notebooks/02_chunking.ipynb
      ```

      - Run the embedding and indexing notebook to create embeddings and build the FAISS index.

      ```bash
      notebooks/03_embedding_and_indexing.ipynb
      ```

      - Run the RAG core logic notebook to implement the retrieval-augmented generation (RAG) approach.

      ```bash
      notebooks/04_rag_core_logic.ipynb
      ```

3. **View curves**
   - Generated plots will be available in the `plots/` directory or as specified by your script parameters.

4. **Scripts and Tests**
   - Explore `scripts/` and `tests/` directory to interact with script and test suits.

5. **Interactive APP**
   - Features:
      - Input box for natural-language questions
      - Answer powered by RAG + semantic retrieval
      - Source chunks displayed beneath answer
      - Dropdown filters for product and issue category
      - Conversation history retention for up to 3 follow-up questions
      - Optional streaming with token-by-token output

   ```bash
   python app.py
   ```

6. **Code Quality and Linting**
    This project uses pre-commit hooks to automatically format and lint `.py` and `.ipynb` files using:

    |Tool       | Purpose                                       |
    |:----------|-----------------------------------------------|
    | Black     |Enforces consistent code formatting            |
    | isort     |Sorts and organises import statements          |
    | Flake8    |Lints Python code for style issues             |
    | nbQA      |Runs Black, isort, and Flake8 inside notebooks |

    ``` bash
    # Format and lint all scripts and notebooks
    pre-commit run --all-files
    ```

---

## Insights

### EDA & Chunking

![Complaints_Narratives](plots/01_eda/Distribution%20of%20Complaints%20With%20vs%20Without%20Narratives.png)

Less than half of complaints included narratives, highlighting the need for improved data collection methods.

---

![Complaints_Product](plots/01_eda/Distribution%20of%20Complaints%20by%20Product.png)

The distribution of complaints by product shows significant variation, indicating areas for targeted improvement.

---

![Complaints_Length](plots/01_eda/Distribution%20of%20Complaint%20Lengths.png)

The length of complaints varies widely, with some being very brief and others quite detailed. This affects the effectiveness of automated analysis.

---

![Complaint_Distribution](plots/02_chunking_embedding/Distribution%20of%20Chunk%20Lengths.png)

The distribution of chunk lengths shows that most chunks are relatively short, with a few outliers that are much longer. This could affect the performance of the embedding model.

---

![Complaint_Distribution_by_Product](plots/02_chunking_embedding/Distribution%20of%20Chunk%20Lengths%20for%20Product%20Types.png)

The distribution of chunk lengths for different product types reveals insights into how complaints are segmented and informs targeted improvements.

---

### RAG Implementation

- The evaluation of the RAG system was conducted using a set of predefined questions and manually scored answers.
- Evaluation Table located at `data/evaluation/rag_evaluation.csv` and `data/evaluation/rag_evaluation.md`. It includes:
      - Questions used (20)
      - Answers generated
      - Source documents pulled
      - Manual score (1–5)
      - Commentary

| Question                                      | Generated Answer (Summary)                                                     | Quality Score | Commentary                                                                 |
|----------------------------------------------|---------------------------------------------------------------------------------|---------------|----------------------------------------------------------------------------|
| Why are users unhappy with payday loans?     | Misleading practices and hidden fees trap users in debt cycles.                 | 5             | Strong abstraction across sources; captures core sentiment fluently.      |
| Do customers report unexpected fees?         | Fees deducted unexpectedly from paychecks, contradicting loan terms.            | 4             | Clear answer, but closely mirrors one complaint without summarising.      |
| What are the most common credit card issues? | Billing disputes, fraud, credit limit reductions, and privacy concerns.         | 4             | Informative, but lifted directly from a retrieved source.                 |
| Are savings accounts being frozen?           | Account freezes cause financial stress; users demand clear explanations.        | 4             | Accurate but lacks synthesis; echoes one complaint nearly verbatim.       |
| Are users misled about promotional offers?   | Promotions are unclear or contradictory, leading to unexpected charges.         | 5             | Well-summarised across multiple complaints; captures systemic issue.      |
| Do users face refund delays?                 | Refunds are delayed without clear updates, violating consumer protection norms. | 5             | Strong generalisation; reflects regulatory framing and user frustration.  |

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

Make sure to follow best practices for version control, testing, and documentation.

---

## Project Status

Final submission merged. Checkout the [commit history](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/commits?author=nuhaminae).
