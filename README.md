# Intelligent Complaint Analysis for Financial Services

[![CI](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/actions/workflows/CI.yml)

## Overview
Intelligent Complaint Analysis for Financial Services is an advanced platform tailored for the automatic analysis, categorisation, and reporting of customer complaints within the financial sector. The system aims to streamline the identification of key issues, sentiment, and trends in customer feedback by leveraging natural language processing (NLP) and machine learning techniques. Ultimately the complaint analysis system helps financial institutions to improve their service quality and regulatory compliance.

## Key Features
- Explaratory data analysis
- Customisable text chunking strategy with overlap control. 
- Embedding via sentence-transformers/all-MiniLM-L6-v2.
- Semantic search over chunked complaint narratives using ChromaDB
- RAG pipeline for question answering with context-aware LLM prompts.
- Interactive UI (Gradio/Streamlit) for non-technical users.
- Persistent vector store with metadata for traceability.
- CI-integrated test suite for reproducibility and robustness

## Table of Contents
- [Project Background](#project-background)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Project Status](#project-status)

## Project Background
The financial services industry faces a high volume of customer complaints daily, many of which contain crucial insights into systemic issues, regulatory risks, or opportunities for improvement. Manual analysis is labour-intensive and prone to oversight. 

This project aims to reduce that time from days to minutes by building a RAG-powered chatbot that retrieves relevant complaint excerpts and generates grounded answers. The system is designed to support internal teams in making faster, evidence-based decisions.

## Data Sources
The project uses real-world complaint data from the Consumer Financial Protection Bureau (CFPB). Each record includes:

- A free-text narrative from the consumer
- Product and company metadata
- Submission dates and issue labels

The dataset was filtered to include only the five target products (Credit card, Personal loan, Buy Now, Pay Later (BNPL), Savings account, Money transfers) and cleaned to remove empty narratives.

## Project Structure
The repository is organised as follows:
```
├──.dvc/
├──.github/workflows
├── data/
│   ├── raw/                         # Original CFPB dataset
│   └── processed/                   # Processed datase
│       ├── filtered_complaints.csv      
|       └── chunked_complaints.csv
├── notebooks/                       # Notebooks
|   ├──01_eda_preprocessing.ipynb
|   ├──02_1_chunking.ipynb
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

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you’d like to change.

Make sure to follow best practices for version control, testing, and documentation.

## Project Status
This project is currently in development. Checkout the commit history [here](https://github.com/nuhaminae/Intelligent-Complaint-Analysis-for-Financial-Services/commits?author=nuhaminae). 
