# test_01_preprocess_data

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import os
import sys
import tempfile
import unittest.mock as mock
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your EDA class
# trunk-ignore(ruff/E402)
from scripts._01_preprocess_data import EDA


@pytest.fixture
def dummy_data():

    os.makedirs(os.path.join("data", "temp"), exist_ok=True)
    np.random.seed(42)  # for reproducibility

    df = pd.DataFrame(
        {
            "Date received": pd.date_range("2024-01-01", periods=200, freq="h"),
            "Product": np.random.choice(
                [
                    "Credit card",
                    "Personal loan",
                    "Buy Now, Pay Later (BNPL)",
                    "Savings account",
                    "Money transfers",
                    "Mortgage",
                    "Debt collection",
                ],
                size=200,
            ),
            "Sub-product": np.random.choice(
                ["Rewards card", "Installment loan", "Online savings"], size=200
            ),
            "Issue": np.random.choice(
                ["Billing dispute", "Incorrect information", "Account closure"],
                size=200,
            ),
            "Sub-issue": np.random.choice(
                ["Late fee", "Unauthorized transaction", "Wrong balance"], size=200
            ),
            "Consumer complaint narrative": np.random.choice(
                [
                    "I am writing to file a complaint about my credit card.",
                    "The loan terms were not clearly explained.",
                    "Money was transferred to the wrong account.",
                    "I was charged twice for the same transaction.",
                    "My savings account was closed without notice.",
                ],
                size=200,
            ),
            "Company public response": np.random.choice(
                ["Company has responded", "No response"], size=200
            ),
            "Company": np.random.choice(
                ["Citibank", "Wells Fargo", "Chime", "Kifiya"], size=200
            ),
            "State": np.random.choice(["CA", "NY", "TX", "GA", "WA"], size=200),
            "Zip code": np.random.randint(10000, 99999, size=200).astype(str),
            "Tags": np.random.choice(["Older American", "Servicemember", ""], size=200),
            "Consumer consent provided?": np.random.choice(
                ["Yes", "No", "Unknown"], size=200
            ),
            "Submitted via": np.random.choice(["Web", "Phone", "Email"], size=200),
            "Date sent to company": pd.date_range("2024-01-01", periods=200, freq="h"),
            "Company response to consumer": np.random.choice(
                ["Closed with explanation", "In progress", "Closed"], size=200
            ),
            "Timely response?": np.random.choice(["Yes", "No"], size=200),
            "Consumer disputed?": np.random.choice(["Yes", "No", "Unknown"], size=200),
            "Complaint ID": np.arange(1000000, 1000200),
        }
    )

    file_path = os.path.join("data", "temp", "temp.csv")
    df.to_csv(file_path, index=False)
    return file_path


# Create an eda instance
def test_eda_instance_creation(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)
    assert eda_instance is not None


# Test load df method
def test_load_df(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    # Assert that df_raw is loaded and is not None
    assert eda_instance.df_raw is not None

    # Assert shape
    assert eda_instance.df_raw.shape == (200, 18)

    # Assert column names are title-cased
    assert all(col == col.title() for col in eda_instance.df_raw.columns)

    # Assert the column names and their case
    expected_columns = [
        "Date Received",
        "Product",
        "Sub-Product",
        "Issue",
        "Sub-Issue",
        "Consumer Complaint Narrative",
        "Company Public Response",
        "Company",
        "State",
        "Zip Code",
        "Tags",
        "Consumer Consent Provided?",
        "Submitted Via",
        "Date Sent To Company",
        "Company Response To Consumer",
        "Timely Response?",
        "Consumer Disputed?",
        "Complaint Id",
    ]
    assert eda_instance.df_raw.columns.tolist() == expected_columns


# Test load df method with invalid path
def test_load_df_with_invalid_path():
    eda = EDA(df_path="nonexistent.csv")
    assert eda.df_raw is None


# Test clean narrative method
def test_clean_narrative():
    raw_text = "Dear CFPB, I am writing to file a complaint about my CREDIT CARD being charged twice!"
    cleaned = EDA.clean_narrative(raw_text)
    assert "credit" in cleaned and "card" in cleaned
    assert "dear" not in cleaned and "cfpb" not in cleaned


# Test visualise complaint method
def test_visualise_complaint(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    plot_name = "Distribution of Complaints by Product.png"

    with (
        mock.patch("matplotlib.pyplot.show"),
        mock.patch("matplotlib.pyplot.savefig") as mock_savefig,
    ):

        eda_instance.visualise_complaint()

        # Assert that savefig was called
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args

        # Ensure the argument is a string and contains the expected filename
        assert isinstance(args[0], str), f"Expected a string path, got: {type(args[0])}"
        assert (
            os.path.basename(args[0]) == plot_name
        ), f"Expected filename '{plot_name}', got: {os.path.basename(args[0])}"


# Test isualise complaint method with invalid df path
def test_visualise_complaint_without_df():
    eda = EDA(df_path=None)
    eda.df_raw = None
    with mock.patch("builtins.print") as mock_print:
        result = eda.visualise_complaint()
        assert result is None
        printed = [
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        ]
        assert any("DataFrame not loaded" in msg for msg in printed)


# Test visualise complaint length method
def test_visualise_complaint_length(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    plot_name = "Distribution of Complaint Lengths.png"
    #expected_plot_path = os.path.join(plot_dir, plot_name)

    with (
        mock.patch("matplotlib.pyplot.show"),
        mock.patch("matplotlib.pyplot.savefig") as mock_savefig,
    ):

        eda_instance.visualise_complaint_length()

        # Assert that savefig was called
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args

        # Ensure the argument is a string and contains the expected filename
        assert isinstance(args[0], str), f"Expected a string path, got: {type(args[0])}"
        assert (
            os.path.basename(args[0]) == plot_name
        ), f"Expected filename '{plot_name}', got: {os.path.basename(args[0])}"


# Test complaints narrative method
def test_complaints_narative(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    plot_name = "Distribution of Complaints With vs Without Narratives.png"

    with (
        mock.patch("matplotlib.pyplot.show"),
        mock.patch("matplotlib.pyplot.savefig") as mock_savefig,
        mock.patch("builtins.print") as mock_print,
    ):

        eda_instance.complaints_narative()

        # Assert that savefig was called
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert plot_name in os.path.basename(args[0])

        # Flatten all print call arguments into strings
        printed_texts = [
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        ]

        # Check that expected phrases appear in the printed output
        assert any("Total complaints: 200" in text for text in printed_texts)
        assert any("Complaints with narratives: 200" in text for text in printed_texts)
        assert any("Complaints without narratives: 0" in text for text in printed_texts)


# Test filter products method
def test_filter_products(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    # Get the initial number of rows
    initial_rows = eda_instance.df_raw.shape[0]

    # Call the method under test
    eda_instance.filter_products()

    # Assert that the df attribute is set
    assert eda_instance.df is not None

    # Assert that the number of rows in the filtered DataFrame is less than or equal to the original
    assert eda_instance.df.shape[0] <= initial_rows

    # Define the expected unique products after filtering
    expected_products = [
        "Credit card",
        "Credit card or prepaid card",  # credit card
        "Payday loan, title loan, or personal loan",  # personal loan
        "Payday loan, title loan, personal loan, or advance loan",  # personal loan
        "Money transfers",  # money transfer
        "Money transfer, virtual currency, or money service",  # money transfer
        "Checking or savings account",  # savings account
        "Payday loan",  # Payday loan
        "Other financial service",  # other financial service
    ]

    # Assert that the unique values in the 'Product' column are within the expected list
    unique_filtered_products = eda_instance.df["Product"].unique().tolist()
    for product in unique_filtered_products:
        assert product in expected_products


# Test missing values method
def test_missing_values(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    # Inject missing values into a copy of the DataFrame
    eda_instance.df = eda_instance.df_raw.copy()
    eda_instance.df.loc[0, "Consumer Complaint Narrative"] = np.nan
    eda_instance.df.loc[1, "State"] = np.nan
    eda_instance.df.loc[2, "Tags"] = np.nan

    with mock.patch("builtins.print") as mock_print:
        eda_instance.missing_values()

        # Check that the narrative column has no missing values
        assert eda_instance.df["Consumer Complaint Narrative"].isna().sum() == 0

        # Check that no missing values remain in the DataFrame
        assert eda_instance.df.isna().sum().sum() == 0

        # Check that "Unknown" appears in the expected columns
        assert "Unknown" in eda_instance.df["State"].values
        assert "Unknown" in eda_instance.df["Tags"].values

        # Check that expected print messages were triggered
        printed = [
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        ]
        assert any(
            'Rows with missing "Consumer Complaint Narrative" have been dropped.' in msg
            for msg in printed
        )
        assert any(
            'Columns with with missing values are filled with "Unknown"' in msg
            for msg in printed
        )


# Test normalise text method
def test_normalise_text(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    # Ensure the DataFrame is loaded and assigned
    eda_instance.df = eda_instance.df_raw.copy()

    # Inject a known narrative for controlled testing
    eda_instance.df.loc[0, "Consumer Complaint Narrative"] = (
        "Dear CFPB, I am writing to file a complaint about my CREDIT CARD being charged twice!"
    )

    with mock.patch("builtins.print") as mock_print:
        eda_instance.normalise_text()

        # Check that the new column was created
        assert "Clean Consumer Complaint Narrative" in eda_instance.df.columns

        # Check that the text was normalized (lowercased, cleaned, etc.)
        cleaned_text = eda_instance.df.loc[0, "Clean Consumer Complaint Narrative"]
        assert "credit" in cleaned_text
        assert "card" in cleaned_text
        assert "charged" in cleaned_text
        assert "dear" not in cleaned_text  # assuming boilerplate removal works

        # Check that the print message was triggered
        printed = [
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        ]
        assert any(
            "Text under 'Consumer Complaint Narrative' column are normalised." in msg
            for msg in printed
        )


# Test save df method
def test_save_df(dummy_data):
    file_path = dummy_data
    plot_dir = tempfile.mkdtemp()
    processed_dir = tempfile.mkdtemp()
    eda_instance = EDA(df_path=file_path, plot_dir=plot_dir, processed_dir=processed_dir)

    # Assign a processed DataFrame
    eda_instance.df = eda_instance.df_raw.copy()

    filename = "test_filtered_complaints.csv"
    expected_path = os.path.join(processed_dir, filename)

    with (
        mock.patch("builtins.print") as mock_print,
        mock.patch("IPython.display.display"),
    ):

        eda_instance.save_df(filename=filename)

        # Check that the file was created
        assert os.path.exists(expected_path)

        # Check that the DataFrame was saved with sorted columns
        saved_df = pd.read_csv(expected_path)
        assert saved_df.columns.tolist() == sorted(saved_df.columns.tolist())

        # Check that print statements were triggered
        printed = [
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        ]
        assert any("Processed DataFrame saved to:" in msg for msg in printed)
        assert any("DataFrame Head:" in msg for msg in printed)
        assert any("Processed DataFrame shape:" in msg for msg in printed)


# Test save df method with invalid df path
def test_save_df_without_df():
    eda = EDA(df_path=None, plot_dir=".", processed_dir=".")
    eda.df = None
    with mock.patch("builtins.print") as mock_print:
        result = eda.save_df()
        assert result is None
        printed = [
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        ]
        assert any("No processed DataFrame found" in msg for msg in printed)
