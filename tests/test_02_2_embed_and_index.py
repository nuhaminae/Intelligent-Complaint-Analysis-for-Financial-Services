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
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")


# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your Chunking class
from scripts._02_2_embed_and_index import EmbeddingIndexer

@atexit.register
def cleanup_temp_data():
    shutil.rmtree(os.path.join('data', 'temp'), ignore_errors=True)

@pytest.fixture
def dummy_data():
    """
    Creates a temporary CSV file with chunked complaint data.
    """
    os.makedirs(os.path.join('data', 'temp'), exist_ok=True)
    np.random.seed(42)  # for reproducibility

    # Sample narratives
    narratives = np.random.choice([
        "I am writing to file a complaint about my credit card.",
        "The loan terms were not clearly explained.",
        "Money was transferred to the wrong account.",
        "I was charged twice for the same transaction.",
        "My savings account was closed without notice."
    ], size=200)

    # Cleaned narratives
    cleaned_narratives = [n.lower().strip().replace(".", "").replace(",", "") for n in narratives]

    # Treat each cleaned narrative as a single chunk
    chunks = cleaned_narratives
    chunk_lengths = [len(c) for c in chunks]
    
    # Complaint lengths (in words)
    complaint_lengths = [len(n.split()) for n in cleaned_narratives]

    df = pd.DataFrame({
        'Chunk': chunks,
        'Chunk Length': chunk_lengths,
        'Company': np.random.choice(['Citibank', 'Wells Fargo', 'Chime', 'Kifiya'], size=200),
        'Company Public Response': np.random.choice(['Company has responded', 'No response'], size=200),
        'Company Response To Consumer': np.random.choice(['Closed with explanation', 'In progress', 'Closed'], size=200),
        'Complaint ID': np.arange(1000000, 1000200),
        'Complaint Length': complaint_lengths,
        'Consumer Complaint Narrative': narratives,
        'Consumer Complaint Narrative Clean': cleaned_narratives,
        'Consumer Consent Provided?': np.random.choice(['Consent Provided', 'No', 'Unknown'], size=200),
        'Consumer Disputed?': np.random.choice(['Yes', 'No', 'Unknown'], size=200),
        'Date Received': pd.date_range('2024-01-01', periods=200, freq='h'),
        'Date Sent To Company': pd.date_range('2024-01-01', periods=200, freq='h'),
        'Issue': np.random.choice(['Billing dispute', 'Incorrect information', 'Account closure'], size=200),
        'Product': np.random.choice(["Credit card", "Personal loan", "Buy Now, Pay Later (BNPL)", "Savings account", "Money transfers", "Mortgage", "Debt collection"], size=200),
        'State': np.random.choice(['CA', 'NY', 'TX', 'GA', 'WA'], size=200),
        'Sub-Issue': np.random.choice(['Late fee', 'Unauthorized transaction', 'Wrong balance'], size=200),
        'Sub-Product': np.random.choice(['Rewards card', 'Installment loan', 'Online savings'], size=200),
        'Submitted Via': np.random.choice(['Web', 'Phone', 'Email'], size=200),
        'Tags': np.random.choice(['Older American', 'Servicemember', ''], size=200),
        'Timely Response?': np.random.choice(['Yes', 'No'], size=200),
        'Zip code': np.random.randint(10000, 99999, size=200).astype(str)
    })

    file_path = os.path.join('data', 'temp', 'chunked_temp.csv')
    df.to_csv(file_path, index=False)
    return file_path
    
# Create an EmbeddingIndexer instance    
def test_embedding_instance_creation(dummy_data):
    file_path = dummy_data
    vector_store_dir = tempfile.mkdtemp()
    embedding_instance = EmbeddingIndexer(df_chunks_path = file_path, 
                                        vector_store_dir = vector_store_dir)
    assert embedding_instance is not None

# Test load chunks method
def test_load_chunks(dummy_data):
    file_path = dummy_data
    vector_store_dir = tempfile.mkdtemp()
    embedding_instance = EmbeddingIndexer(df_chunks_path = file_path, 
                                        vector_store_dir = vector_store_dir)
    
    # Load data
    df_chunks = embedding_instance.load_chunks()

    assert df_chunks is not None, "DataFrame should not be None after loading"
    assert len(df_chunks) == 200, "Expected 200 rows in dummy data"
    assert "Chunk" in df_chunks.columns, "'Chunk' column should be present"
    assert "Consumer Complaint Narrative Clean" in df_chunks.columns, "'Consumer Complaint Narrative Clean' column should be present"
    assert df_chunks["Consumer Complaint Narrative Clean"].notnull().all(), "Cleaned narratives should not contain nulls"

# Test index chunks method
def test_index_chunks(dummy_data):
    file_path = dummy_data
    vector_store_dir = tempfile.mkdtemp()
    embedding_instance = EmbeddingIndexer(df_chunks_path = file_path, 
                                        vector_store_dir = vector_store_dir)
    # Load the data
    embedding_instance.load_chunks()
    # Run indexing
    embedding_instance.index_chunks(batch_size=50)

    #  Check that the collection is populated
    count = embedding_instance.collection.count()
    assert count == 200, f"Expected 200 indexed chunks, but got {count}"

# Test search chunks method
def test_search_chunks(dummy_data):
    file_path = dummy_data
    vector_store_dir = tempfile.mkdtemp()
    embedding_instance = EmbeddingIndexer(df_chunks_path=file_path, vector_store_dir=vector_store_dir)
    embedding_instance.load_chunks()
    embedding_instance.index_chunks(batch_size=50)

    results = embedding_instance.search_chunks(query="unauthorized charges", n_results=5)
    assert results is not None, "Search should return results"
    assert "documents" in results, "Search result should contain 'documents'"
    assert len(results["documents"][0]) > 0, "Should return at least one document"

# Test format search results method
def test_format_search_results(dummy_data):
    file_path = dummy_data
    vector_store_dir = tempfile.mkdtemp()
    embedding_instance = EmbeddingIndexer(df_chunks_path=file_path, vector_store_dir=vector_store_dir)
    embedding_instance.load_chunks()
    embedding_instance.index_chunks(batch_size=50)

    results = embedding_instance.search_chunks(query="loan terms", n_results=3)
    df = embedding_instance.format_search_results(results)
    assert isinstance(df, pd.DataFrame), "Formatted result should be a DataFrame"
    assert "Chunk ID" in df.columns, "Formatted DataFrame should contain 'Chunk ID'"
    assert "Chunk Text" in df.columns, "Formatted DataFrame should contain 'Chunk Text'"

# Test run batch queries grouped method
def test_run_batch_queries_grouped(dummy_data):
    file_path = dummy_data
    vector_store_dir = tempfile.mkdtemp()
    embedding_instance = EmbeddingIndexer(df_chunks_path=file_path, vector_store_dir=vector_store_dir)
    embedding_instance.load_chunks()
    embedding_instance.index_chunks(batch_size=50)

    query_product_pairs = [
        ("unauthorized charges", "Credit card", "Credit Cards"),
        ("loan repayment issues", "Personal loan", "Personal Loans"),
        ("account closed", "Savings account", "Savings Accounts")
    ]

    results = embedding_instance.run_batch_queries_grouped(query_product_pairs, n_results=2)
    assert isinstance(results, dict), "Results should be a dictionary"
    assert all(isinstance(v, pd.DataFrame) for v in results.values()), "Each result should be a DataFrame"