# _07_business_implication.py

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from rapidfuzz import fuzz


class BusinessRisk:
    def __init__(self, processed_path, processed_dir, plot_dir):
        """
        Initialises the BusinessRisk class.

        Args:
            processed_path (str): Path to the processed data file.
            processed_dir (str): Directory where processed DataFrames are stored.
            plot_dir (str): Directory where plots are saved.
        """
        self.processed_path = processed_path
        self.processed_dir = processed_dir
        self.plot_dir = plot_dir

        self.risk_buckets = {
            "Credit Reporting Errors": [
                "incorrect information",
                "credit report",
                "account status",
                "credit score",
            ],
            "Debt Collection Errors": [
                "debt not owed",
                "debt collection",
                "harassment",
                "threats",
            ],
            "Loan Servicing Problems": [
                "lender",
                "vehicle",
                "property",
                "repossession",
                "foreclosure",
            ],
            "Fraud & Unauthorized Use": [
                "identity theft",
                "unauthorized",
                "lost or stolen",
                "fraud",
            ],
            "Billing Disputes": ["fees", "charges", "billing", "overdraft", "interest"],
            "Investigation Failures": [
                "investigation",
                "response",
                "resolution",
                "company failed",
            ],
            "Customer Service Issues": [
                "communication",
                "customer service",
                "rude",
                "unhelpful",
            ],
            "Access & Account Issues": [
                "login",
                "access",
                "account locked",
                "technical issue",
            ],
        }

        self.cost_estimates = {
            "Credit Reporting Errors": 200,
            "Debt Collection Errors": 400,
            "Loan Servicing Problems": 600,
            "Fraud & Unauthorized Use": 800,
            "Billing Disputes": 100,
            "Investigation Failures": 150,
            "Customer Service Issues": 75,
            "Access & Account Issues": 50,
            "Uncategorized": 100,
        }

        self.df = None

        # Create output directories if they do not exist
        if self.plot_dir is not None and not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        if self.processed_dir is not None and not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        print("🧪 Running Business Exposure Analysis Pipeline...\n")

        if self.processed_path:
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

        if not hasattr(self, "processed_path") or self.processed_path is None:
            raise FileNotFoundError("⚠️ DataFrame path is None or invalid.")

        try:
            self.df = pd.read_csv(
                self.processed_path,
                dtype={"Consumer Complaint Narrative": str},
                low_memory=False,
            )
            print(
                f"📥 DataFrame loaded from {self.safe_relpath(self.processed_path)}.\n"
            )
            print(f"🔹 DataFrame columns:\n {self.df.columns.tolist()}")
        except FileNotFoundError as e:
            print(
                f"⚠️ Error loading DataFrame from \
                    \
                    {self.safe_relpath(self.processed_path)}: {e}"
            )
            raise e

    def fuzzy_map_issue(self, issue, buckets):
        """
        Map an issue to a risk category using fuzzy matching.

        Args:
            issue (str): The issue description.
            buckets (dict): A dictionary of risk categories and
                            their associated keywords.

        Returns:
            str: The name of the risk category the issue is mapped to.
        """
        best_score = 0
        best_bucket = "Uncategorized"
        for bucket, keywords in buckets.items():
            for keyword in keywords:
                score = fuzz.partial_ratio(issue.lower(), keyword.lower())
                if score > best_score:
                    best_score = score
                    best_bucket = bucket
        return best_bucket

    def map_issues_to_risk_categories(self):
        """
        Map issues in the DataFrame to predefined risk categories.

        Returns:
            pd.DataFrame: The updated DataFrame with risk categories.
        """
        if "Count" not in self.df.columns:
            self.df["Count"] = 1  # Default to 1 if not pre-aggregated

        self.df["Risk_Category"] = self.df["Issue"].apply(
            lambda x: self.fuzzy_map_issue(x, self.risk_buckets)
        )
        self.df.groupby("Risk_Category")["Issue"].count().sort_values(ascending=False)

        self.df["Estimated_Cost"] = self.df["Risk_Category"].map(self.cost_estimates)
        self.df["Total_Exposure"] = self.df["Count"] * self.df["Estimated_Cost"]

        category_counts = (
            self.df.groupby("Risk_Category")["Issue"]
            .count()
            .sort_values(ascending=False)
        )
        print(f"\n🔹 Total number of issues: {self.df['Issue'].count()}")
        display(category_counts)

    def get_exposure_summary(self):
        """
        Get a summary of the estimated financial exposure by risk category.

        Returns:
            pd.Series: A series containing the total estimated exposure
                        for each risk category.
        """
        self.exposure_summary = (
            self.df.groupby("Risk_Category")["Total_Exposure"]
            .sum()
            .sort_values(ascending=False)
        )
        return self.exposure_summary

    def plot_exposure_summary(self):
        """
        Plot the estimated financial exposure by risk category.
        """
        self.exposure_summary = self.get_exposure_summary()

        plt.figure(figsize=(10, 4))
        sns.barplot(
            x=self.exposure_summary.values,
            y=self.exposure_summary.index,
            hue=self.exposure_summary.index,
            palette="Set1",
        )
        plt.title("Estimated Financial Exposure by Risk Category")
        plt.xlabel("Estimated Exposure ($)")
        plt.ylabel("Risk Category")
        plt.grid(True)

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir,
                "Estimated Financial Exposure by Risk Category.png",
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def save_df(self, filename="complaints_analysis.csv"):
        """
        Saves the processed DataFrame to the specified directory.

        Args:
            filename (str, optional): The name of the output file.
                                        Defaults to 'complaints_analysis.csv'.
        """

        if not hasattr(self, "df") or self.df is None:
            print("\n⚠️ No processed DataFrame. Run preprocessing before saving.")
            return

        # Sort and save processed DataFrame to CSV
        self.df = self.df[sorted(self.df.columns)]

        df_name = os.path.join(self.processed_dir, filename)
        self.df.to_csv(df_name, index=False)

        print(f"\n💾 Processed DataFrame saved to: {self.safe_relpath(df_name)}")

        print("\n🔹 DataFrame Head:")
        display(self.df.head())
        print(f"\n🔹Processed DataFrame shape: {self.df.shape}")
        print(f"\n🔹 Processed DataFrame columns:\n {self.df.columns.tolist()}\n")

    def run_business_exposure_pipeline(self):
        """
        Run the business exposure analysis pipeline.
        """
        self.map_issues_to_risk_categories()
        self.plot_exposure_summary()
        self.save_df()
