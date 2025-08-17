# _02_chunking.py

import os

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from IPython.display import display
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")


class Chunking:
    def __init__(self, df_path, plot_dir=None, processed_dir=None):
        """
        Initiate Chunking class from DataFrame path.

        Args:
            df_path (str): The path to the DataFrame file (e.g., CSV).
            plot_dir (str, optional): The directory to save plots. Defaults to None.
            processed_dir (str, optional): The directory to save processed DataFrames.
                                    Defaults to None.
        """

        self.df_path = df_path
        self.plot_dir = plot_dir
        self.processed_dir = processed_dir

        # Create output directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        print("🧪 Running full Chunking pipeline...\n")

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
        Load DataFrame
        """
        if not hasattr(self, "df_path") or self.df_path is None:
            raise FileNotFoundError("⚠️ DataFrame path is None or invalid.")

        try:
            self.df = pd.read_csv(self.df_path)
            if "Date Received" in self.df.columns:
                self.df["Date Received"] = pd.to_datetime(
                    self.df["Date Received"], errors="coerce"
                )
            if "Date Sent To Company" in self.df.columns:
                self.df["Date Sent To Company"] = pd.to_datetime(
                    self.df["Date Sent To Company"], errors="coerce"
                )

            print(f"📥 DataFrame loaded from  {self.safe_relpath(self.df_path)}")

        except FileNotFoundError as e:
            print(f"⚠️ Error loading DataFrame : {e}")
            raise e

        return self.df

    # ----------------------------- Chunking ----------------------------- #

    def sentence_chunker(self, text, max_words=100):
        """
        Chunk text into sentence-aware segments with a max word count.

        Args:
            text (str): The input narrative.
            max_words (int): Maximum number of words per chunk.

        Returns:
            List[str]: List of sentence-based chunks.
        """
        if pd.isna(text):
            return []

        sentences = sent_tokenize(text)
        chunks, current_chunk = [], []
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk, word_count = [], 0
            current_chunk.append(sentence)
            word_count += len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_narrative(self, max_words=100):
        """
        Chunk narratives using sentence-aware chunking and explode into multiple rows.
        Saves the chunked DataFrame to self.df_chunks.
        """
        if hasattr(self, "df") and self.df is not None:
            if "Clean Narrative" not in self.df.columns:
                print("\n⚠️ Error: 'Clean Narrative' column not found.")
                return None

            # Apply sentence-aware chunking
            self.df["Chunk"] = self.df["Clean Narrative"].apply(
                lambda x: self.sentence_chunker(x, max_words=max_words)
            )

            # Explode into multiple rows
            self.df_chunks = self.df.explode("Chunk").reset_index(drop=True)

            print(f"\n✅ Sentence-aware chunking complete with max_words={max_words}.")
            return self.df_chunks

        else:
            print(
                "\n⚠️ No DataFrame found. Please ensure DataFrame is loaded correctly."
            )
            return None

    def plot_chunk_lengths(self):
        """
        Plot distribution of chunk lengths.
        """
        if self.df_chunks is None:
            print("\n⚠️ Chunked DataFrame missing. Run chunk_narrative() first.")
            return

        self.df_chunks["Chunk Length"] = self.df_chunks["Chunk"].apply(len)
        self.df_chunks["Chunk Length"] = self.df_chunks["Chunk Length"].astype(int)

        # Sort Columns
        self.df_chunks = self.df_chunks[sorted(self.df_chunks.columns)]

        print("\nChunked DataFrame head:")
        display(self.df_chunks.head())

        print(f"\n🔹 DataFrame shape after exploding: {self.df_chunks.shape}")
        print(f"🔹 DataFrame shape before exploding: {self.df.shape}")

        print("🔹 Chunked DataFrame summary:\n")
        self.df_chunks.info()

        # Plot the distribution of chunk length
        plt.figure(figsize=(10, 4))
        sns.histplot(
            self.df_chunks["Chunk Length"], bins=100, kde=True, color="skyblue"
        )
        plt.axvline(
            self.df_chunks["Chunk Length"].mean(),
            color="black",
            linestyle="--",
            label="Mean",
        )
        plt.title("Distribution of Chunk Lengths")
        plt.xlabel("Characters per Chunk")
        plt.grid(True)
        plt.tight_layout()

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(self.plot_dir, "Distribution of Chunk Lengths.png")
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

        # Boxplot for chunk lengths by product
        plt.figure(figsize=(10, 10))
        palette = sns.color_palette("Set1", len(self.df_chunks["Product"].unique()))
        sns.boxplot(
            data=self.df_chunks,
            x="Product",
            y="Chunk Length",
            hue="Product",
            palette=palette,
        )
        plt.title("Distribution of Chunk Lengths for Product Types")
        plt.xticks(rotation=45, ha="right")
        plt.grid()

        # Adjust layout and show plot
        plt.tight_layout()
        if self.plot_dir:
            plot_path = os.path.join(
                self.plot_dir, "Distribution of Chunk Lengths for Product Types.png"
            )
            plt.savefig(plot_path)
            print(f"\n💾 Plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    # --------------------------------------------------------------------------------#
    def save_df(self, filename="chunked_complaints.csv"):
        """
        Saves the processed DataFrame to the specified directory.

        Args:
            filename (str, optional): The name of the output file.
            Defaults to 'chunked_complaints.csv'.
        """

        if not hasattr(self, "df_chunks") or self.df_chunks is None:
            print("\n⚠️ No DataFrame found. Please ensure DataFrame is loaded correctly")
            return

        # Create output folder if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        df_name = os.path.join(self.processed_dir, filename)

        # Save chunked DataFrame to CSV
        self.df_chunks.to_csv(df_name, index=False)
        print(f"\n💾 Chunked DataFrame saved to: {self.safe_relpath(self.df_path)}")
        display(self.df_chunks.head())

    # ----- Run full pipeline ----- #
    def chunker(self):
        """
        Runs the full chunking pipeline: data cleaning, chunking,
        and saving processed data.
        """
        self.chunk_narrative()
        self.plot_chunk_lengths()
        self.save_df()
