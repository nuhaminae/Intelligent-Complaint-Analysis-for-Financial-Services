# _01_preprocess_data.py

import os
import re
import unicodedata
import contractions
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display


class EDA:
    def __init__(self, df_path, plot_dir=None, processed_dir=None):
        """
        Initiate EDA class from DataFrame path.

        Args:
            df_path (str): The path to the DataFrame file (e.g., CSV).
            plot_dir (str, optional): The directory to save plots. Defaults to None.
            processed_dir (str, optional): The directory to save processed DataFrames. Defaults to None.
        """

        self.df_path = df_path
        self.plot_dir = plot_dir or "plots"
        self.processed_dir = processed_dir

        self.df_raw = None

        # Create output directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.df_path:
            self.load_df()

    def safe_relpath(self, path, start=None):
        """
        Return a relative path, handling cases where paths are on different drives.

        Args:
            path (str): The path to make relative.
            start (str, optional): The starting directory.
                                    Defaults to current working directory.

        Returns:
            str: The relative path if possible, otherwise the original path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            # Fallback to absolute path if on different drives
            return path

    def load_df(self):
        """
        Load DataFrame and understand the structure of the dataset
            - Number of rows, columns, and data types
        """

        if not hasattr(self, "df_path") or self.df_path is None:
            raise FileNotFoundError("⚠️ DataFrame path is None or invalid.")

        try:
            # self.df_raw = pd.read_csv(self.df_path)
            self.df_raw = pd.read_csv(self.df_path,
                                    dtype={"Consumer complaint narrative": str},
                                    low_memory=False,
                                    )
            if "Date received" in self.df_raw.columns:
                self.df_raw["Date received"] = pd.to_datetime(
                    self.df_raw["Date received"], errors="coerce"
                )
            if "Date sent to company" in self.df_raw.columns:
                self.df_raw["Date sent to company"] = pd.to_datetime(
                    self.df_raw["Date sent to company"], errors="coerce"
                )
            # Capitalise column names for consistency
            self.df_raw.columns = [col.title() for col in self.df_raw.columns]

            print(f"DataFrame loaded successfully from {self.safe_relpath(self.df_path)}")
            print("\n🔹 DataFrame head:")
            display(self.df_raw.head())

            print("\n🔹 DataFrame tail:")
            display(self.df_raw.tail())

            print("\n🔹 DataFrame shape:")
            display(self.df_raw.shape)

            print("\n🔹 DataFrame columns:")
            display(self.df_raw.columns)

            print("\n🔹 DataFrame summary:")
            self.df_raw.info()

        except FileNotFoundError as e:
            print(f"⚠️ Error loading DataFrame from {self.safe_relpath(self.df_path)}: {e}")
            raise e

        return self.df_raw

    # -----------------------------Initial EDA-----------------------------#
    def visualise_complaint(self):
        """
        Visualise the distribution of complaints by product.
        """
        if not hasattr(self, "df_raw") or self.df_raw is None:
            print("⚠️ DataFrame not loaded. Please check initialisation.")
            return None

        # Count complaints by product
        product_counts = self.df_raw["Product"].value_counts()
        print("🧮 Complaint count by product:")
        display(product_counts)

        # Plot the distribution of complaints by product
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=product_counts.index, x=product_counts.values, hue=product_counts.index
        )
        plt.title("Distribution of Complaints by Product")
        plt.xlabel("Number of Complaints")
        plt.ylabel("Product")
        plt.grid()

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, "Distribution of Complaints by Product.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def visualise_complaint_length(self):
        """
        Visualise the length of complaints in the 'Consumer Complaint Narrative' column.
        """

        if not hasattr(self, "df_raw") or self.df_raw is None:
            print("⚠️ DataFrame not loaded. Please check initialisation.")
            return None

        # Calculate the length of each complaint
        self.df_raw["Complaint Length"] = self.df_raw[
            "Consumer Complaint Narrative"
        ].str.len()

        # Plot the distribution of complaint lengths
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df_raw["Complaint Length"], bins=50, kde=True, color="blue")
        plt.title("Distribution of Complaint Lengths")
        plt.xlabel("Length of Complaint")
        plt.ylabel("Frequency")
        plt.grid()

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, "Distribution of Complaint Lengths.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")
            
        plt.show()
        plt.close()

    # -----------------------------Complaints Narrative-----------------------------#
    def complaints_narrative(self):
        """
        Visualise the 'Consumer Complaint Narrative' column.
        """
        if not hasattr(self, "df_raw") or self.df_raw is None:
            print("⚠️ DataFrame not loaded. Please check initialisation.")
            return None

        # Total number of complaints
        total_complaints = len(self.df_raw)

        # Complaints with narratives
        with_narrative = self.df_raw["Consumer Complaint Narrative"].notna().sum()

        # Complaints without narratives
        without_narrative = self.df_raw["Consumer Complaint Narrative"].isna().sum()

        print(f"Total complaints: {total_complaints}")
        print(f"Complaints with narratives: {with_narrative}")
        print(f"Complaints without narratives: {without_narrative}")

        # Visualise the distribution of complaints with and without narratives
        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=["With Narrative", "Without Narrative"],
            y=[with_narrative, without_narrative],
            hue=["With Narrative", "Without Narrative"],
        )
        plt.title("Complaints With vs. Without Narratives")
        plt.ylabel("Number of Complaints")
        plt.grid()
        
        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, "Distribution of Complaints With vs Without Narratives.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    # -----------------------------Filter Dataset-----------------------------#
    def filter_products(self):
        """
        Filter DataFrame to include only relevant products for analysis.
        """
        if hasattr(self, "df_raw") and self.df_raw is not None:
            # Define the products of interest
            products_of_interest = [
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


            # Filter the DataFrame
            self.df = self.df_raw[
                self.df_raw["Product"].isin(products_of_interest)
            ].copy()
            print("🫧 DataFrame is filtered.")

        else:
            print("⚠️ DataFrame not loaded. Please check initialisation.")
            return None

    def missing_values(self):
        """
        Identify missing values and handle them appropriately.
        """
        if hasattr(self, "df") and self.df is not None:
            if self.df.isna().sum().sum() == 0:
                print("🚫 There are no columns with missing values")

            else:
                # Handle missing values in 'Consumer Complaint Narrative'
                if self.df["Consumer Complaint Narrative"].isna().sum() > 0:
                    self.df.dropna(
                        subset=["Consumer Complaint Narrative"], inplace=True
                    )
                    print(
                        '\n🚮 Rows with missing "Consumer Complaint Narrative" have been dropped.'
                    )
                    print(f"🔹 Rows remaining: {self.df.shape[0]}")

                # Identify and fill the rest of the missing values with "Unknown"
                mis_values = self.df.isna().sum()
                missing_cols = mis_values[mis_values > 0].index.tolist()
                self.df[missing_cols] = self.df[missing_cols].fillna("Unknown")
                print(
                    '\n🔹 Columns with with missing values are filled with "Unknown":',
                    missing_cols,
                )

                self.df = self.df.reset_index(drop=True)
        else:
            print("⚠️ DataFrame not loaded. Please check initialisation.")
            return None

    # -----------------------------Normalise Dataset Text-----------------------------#

    def clean_narrative(self, text):
        if pd.isna(text):
            return ""

        # Lowercase and expand contractions
        text = contractions.fix(text.lower())

        # Remove boilerplate
        boilerplate_patterns = [
            r"^i am writing to (file|submit|lodge) a complaint.*?",
            r"^to whom it may concern[:,]?",
            r"^hello[:,]?",
            r"^hi[:,]?",
            r"^dear( [a-z]+)?[:,]?",
            r"^this is regarding.*?",
            r"^i am writing to dispute.*?",
            r"^i would like to report.*?",
            r"^i am reaching out.*?",
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Redact PII
        text = re.sub(r"\b\d{4,}\b", "xxxx", text)
        text = re.sub(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b", "xx/xx/xxxx", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Normalize unicode and collapse whitespace
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalise_text(self):
        """
        Normalise text under 'Consumer Complaint Narrative' columns
        """
        if not hasattr(self, "df") or self.df is None:
            print(
                "\n⚠️ No processed DataFrame found. Please run preprocessing steps before saving."
            )
            return

        self.df["Clean Narrative"] = self.df[
            "Consumer Complaint Narrative"
        ].apply(self.clean_narrative)
        print("\n✅ Normalisation complete.")

    # --------------------------------------------------------------------------------#
    def save_df(self, filename="filtered_complaints.csv"):
        """
        Saves the processed DataFrame to the specified directory.

        Args:
            filename (str, optional): The name of the output file. Defaults to 'filtered_complaints.csv'.
        """

        if not hasattr(self, "df") or self.df is None:
            print(
                "\n⚠️ No processed DataFrame found. Please run preprocessing steps before saving."
            )
            return

        # Sort and save processed DataFrame to CSV
        self.df = self.df[sorted(self.df.columns)]
        
        df_name = os.path.join(self.processed_dir, filename)
        self.df.to_csv(df_name, index=False)

        print(f"\n💾 Processed DataFrame saved to: {self.safe_relpath(df_name)}")

        print("\n🔹 DataFrame Head:")
        display(self.df.head())

        print(f"\n🔹Processed DataFrame shape: {self.df.shape}")
