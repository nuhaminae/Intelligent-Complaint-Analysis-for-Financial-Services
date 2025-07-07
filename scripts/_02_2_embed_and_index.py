import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from packaging import version

if version.parse(chromadb.__version__) < version.parse("0.4.0"):
    raise RuntimeError("Please upgrade ChromaDB to >= 0.4.0 for compatibility.")


class EmbeddingIndexer:
    """
    A class to embed chunked complaint narratives and store them in a ChromaDB vector store.
    """

    def __init__(self, df_chunks_path, vector_store_dir="vector_store/", model_name="all-MiniLM-L6-v2"):
        """
        Initialise the EmbeddingIndexer with paths and model configuration.
        This class is designed for use with ChromaDB's PersistentClient (v0.4.0 or higher),
            which automatically handles persistence to disk. It assumes that:
            
            - The vector store is either empty or isolated (no deduplication is performed).
            - The collection name is fixed as 'complaints_chunks'.
            - The embedding function is compatible with the SentenceTransformer model specified.
            - ChromaDB >= 0.4.0 is installed; earlier versions are not supported.
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
        try:
            self.collection = self.client.get_or_create_collection(
                name="complaints_chunks",
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create or retrieve ChromaDB collection: {e}")


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

    def index_chunks(self, batch_size=500):
        """
        Embed and index the complaint chunks into ChromaDB in batches.
        Skips indexing if the collection already contains data.

        Args:
            batch_size (int): Number of chunks to index per batch. Defaults to 500.
        """
        if self.df_chunks is None:
            raise ValueError("DataFrame not loaded. Run load_chunks() first.")

        # Check if collection already contains data
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"Skipping indexing: collection already contains {existing_count} chunks.")
                print(f"\nVector store saved to: {self.safe_relpath(self.vector_store_dir)}")
                return
        except Exception as e:
            print(f"Warning: Could not check collection count. Proceeding anyway. Error: {e}")

        print(f"\nIndexing {len(self.df_chunks)} chunks into ChromaDB in batches of {batch_size}...")

        documents, metadatas, ids = [], [], []
        added = 0

        for i, row in tqdm(self.df_chunks.iterrows(), total=len(self.df_chunks)):
            chunk_id = f"chunk-{i}"
            documents.append(row["Chunk"])
            metadatas.append(row.to_dict())
            ids.append(chunk_id)

            if len(documents) == batch_size:
                try:
                    self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    added += len(documents)
                except Exception as e:
                    print(f"\nFailed to add batch ending at chunk-{i}: {e}")
                documents, metadatas, ids = [], [], []

        # Add any remaining chunks
        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                added += len(documents)
            except Exception as e:
                print(f"\nFailed to add final batch: {e}")

        print(f"\nIndexing complete. Total chunks added: {added}")
        print(f"\nVector store saved to: {self.safe_relpath(self.vector_store_dir)}")