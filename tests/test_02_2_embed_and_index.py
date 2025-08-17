import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import pytest
import os, sys
import tempfile
import shutil
import atexit
import warnings
from langchain_core.documents import Document
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your Chunking class
from scripts._02_2_embed_and_index import LangchainIndexer

# Tear down dummy data
@atexit.register
def cleanup_temp_data():
    shutil.rmtree(os.path.join('data', 'temp'), ignore_errors=True)

@pytest.fixture
def dummy_chunked_csv():
    """
    Create a temporary CSV with dummy complaint chunks.
    """
    np.random.seed(42)
    
    temp_dir = os.path.join("data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    chunks = np.random.choice([
        "My card was charged incorrectly.",
        "Loan terms were unclear.",
        "Account frozen without reason.",
        "Money transfer failed multiple times.",
        "Unauthorized charges on my account."
    ], size=100)

    df = pd.DataFrame({
        "Chunk": chunks,
        "Product": np.random.choice(["Credit card", "BNPL", "Personal loan"], size=100),
        "Issue": np.random.choice(["Billing dispute", "Delayed response"], size=100),
        "Company": np.random.choice(["Kifiya", "Chime", "Citibank"], size=100),
        "Complaint ID": np.arange(1, 101)
    })

    path = os.path.join(temp_dir, "dummy_chunks.csv")
    df.to_csv(path, index=False)
    
    yield path  #Provide the path to the test

    # Teardown logic after test finishes
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_indexer_initialisation(dummy_chunked_csv):
    temp_dir = tempfile.mkdtemp()
    indexer = LangchainIndexer(df_chunks_path=dummy_chunked_csv, vector_store_dir=temp_dir)
    assert indexer.df_chunks_path == dummy_chunked_csv
    assert os.path.exists(temp_dir)
    assert indexer.embedding_model is not None
    assert indexer.vectorstore is not None

def test_load_chunks(dummy_chunked_csv):
    indexer = LangchainIndexer(df_chunks_path=dummy_chunked_csv)
    indexer.load_chunks()
    assert indexer.df_chunks is not None
    assert "Chunk" in indexer.df_chunks.columns
    assert len(indexer.df_chunks) == 100

def test_index_chunks(dummy_chunked_csv):
    temp_dir = tempfile.mkdtemp()
    indexer = LangchainIndexer(df_chunks_path=dummy_chunked_csv, vector_store_dir=temp_dir)
    indexer.load_chunks()
    indexer.index_chunks(batch_size=25)
    assert indexer.vectorstore._collection.count() > 0

def test_search_function(dummy_chunked_csv):
    temp_dir = tempfile.mkdtemp()
    indexer = LangchainIndexer(df_chunks_path=dummy_chunked_csv, vector_store_dir=temp_dir)
    indexer.load_chunks()
    indexer.index_chunks(batch_size=50)

    results = indexer.search(query="unauthorized charges", k=3)
    assert isinstance(results, list)
    assert all(isinstance(doc, Document) for doc in results)
    assert len(results) <= 3

def test_preview_results_output(dummy_chunked_csv, capsys):
    temp_dir = tempfile.mkdtemp()
    indexer = LangchainIndexer(df_chunks_path=dummy_chunked_csv, vector_store_dir=temp_dir)
    indexer.load_chunks()
    indexer.index_chunks(batch_size=50)
    
    indexer.preview_results(query="loan terms", k=2)
    captured = capsys.readouterr()
    assert "--- Result 1 ---" in captured.out
    assert "Chunk:" in captured.out
