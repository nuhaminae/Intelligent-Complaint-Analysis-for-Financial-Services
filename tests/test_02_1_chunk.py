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
from scripts._02_1_chunk import Chunking

@atexit.register
def cleanup_temp_data():
    shutil.rmtree(os.path.join('data', 'temp'), ignore_errors=True)

@pytest.fixture
def dummy_data():
    """
    Creates a temporary CSV file with sample complaint data.
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

    # Complaint lengths
    complaint_lengths = [len(n.split()) for n in cleaned_narratives]

    df = pd.DataFrame({
        'Clean Consumer Complaint Narrative': cleaned_narratives,
        'Company': np.random.choice(['Citibank', 'Wells Fargo', 'Chime', 'Kifiya'], size=200),
        'Company Public Response': np.random.choice(['Company has responded', 'No response'], size=200),
        'Company Response To Consumer': np.random.choice(['Closed with explanation', 'In progress', 'Closed'], size=200),
        'Complaint ID': np.arange(1000000, 1000200),
        'Complaint Length': complaint_lengths,
        'Consumer Complaint Narrative': narratives,
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

    file_path = os.path.join('data', 'temp', 'filtered_temp.csv')
    df.to_csv(file_path, index=False)
    return file_path

    
# Create an chunking instance    
def test_chunking_instance_creation(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    df_dir = tempfile.mkdtemp()
    chunking_instance = Chunking(df_path=file_path, plot_dir=plot_dir, df_dir=df_dir)
    assert chunking_instance is not None

# Test load df method
def test_load_df(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    df_dir = tempfile.mkdtemp()
    chunking_instance = Chunking(df_path=file_path, plot_dir=plot_dir, df_dir=df_dir)

    assert chunking_instance.df is not None
    assert len(chunking_instance.df) == 200
    assert "Consumer Complaint Narrative Clean" in chunking_instance.df.columns
    assert "Clean Consumer Complaint Narrative" not in chunking_instance.df.columns
    assert chunking_instance.df["Consumer Complaint Narrative Clean"].notnull().all()
    assert pd.api.types.is_datetime64_any_dtype(chunking_instance.df["Date Received"])
    assert pd.api.types.is_datetime64_any_dtype(chunking_instance.df["Date Sent To Company"])


def test_chunk_narrative(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    df_dir = tempfile.mkdtemp()
    chunking_instance= Chunking(df_path=file_path, plot_dir=plot_dir, df_dir=df_dir)

    df_chunks = chunking_instance.chunk_narrative()
    assert df_chunks is not None
    assert "Chunk" in df_chunks.columns
    assert df_chunks["Chunk"].apply(lambda x: isinstance(x, str)).any()

def test_plot_chunk_lengths(dummy_data, tmp_path):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    df_dir = tempfile.mkdtemp()
    chunking_instance= Chunking(df_path=file_path, plot_dir=plot_dir, df_dir=df_dir)

    chunking_instance.chunk_narrative()
    fig = chunking_instance.plot_chunk_lengths()
    assert fig is not None
    plot_file = os.path.join(plot_dir, "Distribution of Chunk Lengths.png")
    assert os.path.exists(plot_file)

def test_save_df(dummy_data, tmp_path):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    df_dir = tempfile.mkdtemp()
    chunking_instance= Chunking(df_path=file_path, plot_dir=plot_dir, df_dir=df_dir)

    chunking_instance.chunk_narrative()
    chunking_instance.save_df("test_output.csv")
    output_file = os.path.join(df_dir, "test_output.csv")
    assert os.path.exists(output_file)
    saved_df = pd.read_csv(output_file)
    assert "Chunk" in saved_df.columns

def test_safe_relpath():
    abs_path = os.path.abspath(__file__)
    rel_path = Chunking.safe_relpath(abs_path)
    assert isinstance(rel_path, str)
