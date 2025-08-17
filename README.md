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
├── insights/                          # Plots and charts for reporting
├── notebooks/                         # Notebooks
├── scripts/                           # Core scripts
├── tests/                             # Unit and pytest
├── .dvcignore                         # Ignore DVC files
├── .flake8                            # Flake8 configuration
├── .gitignore                         # Ignore unnecessary files
├── .pre-commit-config.yaml            # Pre-commit configuration
├── .trunkignore                       # Ignore trunk files
├── .yamllint.yml                      # YAML lint configuration
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
   - Run notebook in `notebooks/` in chronological order.

3. **View curves**
   - Generated plots will be available in the `insights/` directory or as specified by your script parameters.

4. **Scripts and Tests**
   - Explore `scripts/` and `tests/` directory to interact with script and test suits.

5. **Interactive APP**

   ```bash
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

7. **Explore with Notebooks**

    Notebooks are provided for exploratory and iterative development:
    - `notebooks/01_eda_preprocessing.ipynb`
    - `notebooks/02_chunking.ipynb`
    - `notebooks/03_embedding_and_indexing.ipynb`
    - `notebooks/04_rag_core_logic.ipynb`

    Open with Jupyter or VSCode to navigate the workflow interactively.

8. **Code Quality and Linting**
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

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.

Make sure to follow best practices for version control, testing, and documentation.

---

## Project Status

Final submission merged. Checkout the [commit history](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/commits?author=nuhaminae).
