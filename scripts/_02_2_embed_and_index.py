import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

class EmbeddingIndexer:
    """
    A class to embed chunked complaint narratives and store them in a ChromaDB vector store.
    """

    def __init__(self, df_chunks_path, vector_store_dir="vector_store/", model_name="all-MiniLM-L6-v2"):
        """
        Initialise the EmbeddingIndexer with paths and model configuration.

        Args:
            df_chunks_path (str): Path to the chunked DataFrame file (CSV).
            vector_store_dir (str): Directory to persist the ChromaDB vector store.
            model_name (str): Name of the sentence-transformers model to use.
        """
        self.df_chunks_path = df_chunks_path
        self.vector_store_dir = vector_store_dir
        self.model_name = model_name
        self.df_chunks = None

        os.makedirs(self.vector_store_dir, exist_ok=True)

        # Initialise ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.vector_store_dir)


        # Set up embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=f"sentence-transformers/{model_name}"
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="complaints_chunks",
            embedding_function=self.embedding_fn
        )

    @staticmethod
    def safe_relpath(path, start=os.getcwd()):
        """
        Safely compute a relative path, falling back to absolute if needed.

        Args:
            path (str): Target path.
            start (str): Base path to compute relative path from.
        Returns:
            str: Relative or absolute path.
        """
        try:
            return os.path.relpath(path, start)
        except ValueError:
            return path

    def load_chunks(self):
        """
        Load the chunked DataFrame from CSV.

        Returns:
            pd.DataFrame: Loaded DataFrame with chunked complaint narratives.
        """
        rel_df_path = self.safe_relpath(self.df_chunks_path)

        if self.df_chunks_path:
            try:
                self.df_chunks = pd.read_csv(self.df_chunks_path)
                if "Chunk" not in self.df_chunks.columns:
                    raise ValueError("Missing 'Chunk' column in the loaded DataFrame.")
                print(f"\nDataFrame loaded successfully from {rel_df_path}")
                return self.df_chunks
            except Exception as e:
                print(f"Error loading Chunked DataFrame from {rel_df_path}: {e}")
                self.df_chunks = None
        else:
            print(f"Error: File not found at {rel_df_path}")
            self.df_chunks = None

        return self.df_chunks

    def index_chunks(self):
        """
        Embed and index the complaint chunks into ChromaDB.
        Skips chunks that have already been indexed by checking existing IDs.
        """
        if self.df_chunks is None:
            raise ValueError("DataFrame not loaded. Run load_chunks() first.")

        print("Fetching existing IDs from ChromaDB...")
        existing_ids = set()

        try:
            # ChromaDB does not support listing all IDs directly,
            # so fetch in batches using pagination
            offset = 0
            batch_size = 1000
            while True:
                results = self.collection.get(
                    include=["ids"],
                    limit=batch_size,
                    offset=offset
                )
                if not results["ids"]:
                    break
                existing_ids.update(results["ids"])
                offset += batch_size
        except Exception as e:
            print(f"Warning: Could not fetch existing IDs. Proceeding without deduplication. Error: {e}")

        print(f"\nFound {len(existing_ids)} existing chunks. Starting indexing...")

        skipped = 0
        added = 0

        for i, row in tqdm(self.df_chunks.iterrows(), total=len(self.df_chunks)):
            chunk_id = f"chunk-{i}"
            if chunk_id in existing_ids:
                skipped += 1
                continue

            chunk_text = row["Chunk"]
            metadata = row.to_dict()

            try:
                self.collection.add(
                    documents=[chunk_text],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
                added += 1
            except Exception as e:
                print(f"\nFailed to add chunk-{i}: {e}")

        self.client.persist()
        print(f"\nIndexing complete. Added: {added}, Skipped: {skipped}")
        print(f"\nVector store saved to: {self.safe_relpath(self.vector_store_dir)}")
