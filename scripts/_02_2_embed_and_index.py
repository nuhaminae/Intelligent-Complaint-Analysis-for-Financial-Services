import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from packaging import version
from IPython.display import display

if version.parse(chromadb.__version__) < version.parse("0.4.0"):
    raise RuntimeError("Please upgrade ChromaDB to >= 0.4.0 for compatibility.")

class EmbeddingIndexer:
    def __init__(self, df_chunks_path, vector_store_dir="vector_z/", model_name="all-MiniLM-L6-v2"):
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

        # Create directory if it does not exist
        os.makedirs(self.vector_store_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.vector_store_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=f"sentence-transformers/{model_name}"
        )

        try:
            self.collection = self.client.get_or_create_collection(
                name="complaints_chunks",
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create or retrieve ChromaDB collection: {e}")

        print(f"Using model: {self.model_name}\n")
        print(f"Embedding dimension: {self.embedding_fn._model.get_sentence_embedding_dimension()}")

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
        try:
            self.df_chunks = pd.read_csv(self.df_chunks_path)
            if "Chunk" not in self.df_chunks.columns:
                raise ValueError("Missing 'Chunk' column in the loaded DataFrame.")
            print(f"\nDataFrame loaded successfully from {rel_df_path}")
        except Exception as e:
            print(f"Error loading Chunked DataFrame from {rel_df_path}: {e}")
            self.df_chunks = None
        return self.df_chunks

    def index_chunks(self, batch_size=500):
        """
        Embed and index the complaint chunks into ChromaDB in batches.

        Args:
            batch_size (int): Number of chunks to encode per batch. Defaults to 500.
        """
        if self.df_chunks is None:
            raise ValueError("DataFrame not loaded. Run load_chunks() first.")

        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"Skipping indexing: collection already contains {existing_count} chunks.")
                print(f"\nVector store saved to: {self.safe_relpath(self.vector_store_dir)}")
                return
        except Exception as e:
            print(f"Warning: Could not check collection count. Proceeding anyway. Error: {e}")

        print(f"\nEncoding and indexing {len(self.df_chunks)} chunks into ChromaDB...")

        chunks = self.df_chunks["Chunk"].tolist()
        embeddings = self.embedding_fn._model.encode(chunks, batch_size=batch_size, show_progress_bar=True)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"\nEach embedding has {embeddings.shape[1]} dimensions")

        documents = chunks
        metadatas = self.df_chunks.to_dict(orient="records")
        ids = [f"chunk-{i}" for i in range(len(chunks))]

        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
            print(f"\nIndexing complete. Total chunks added: {len(documents)}")
        except Exception as e:
            print(f"\nFailed to add documents to ChromaDB: {e}")

        print(f"\nVector store saved to: {self.safe_relpath(self.vector_store_dir)}")

    def search_chunks(self, query, n_results=3, where=None):
        """
        Search for the most relevant chunks to a query using semantic similarity.

        Args:
            query (str): The search query string.
            n_results (int): Number of top results to return. Defaults to 3.
            where (dict): Optional metadata filter (e.g., {"Product": "Credit card"}).
        Returns:
            dict: ChromaDB query result containing documents, metadata, distances, etc.
        """
        query_embedding = self.embedding_fn._model.encode([query])
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            print(f"Search failed: {e}")
            return None

    def format_search_results(self, results):
        """
        Convert ChromaDB search result dictionary into a readable DataFrame.

        Args:
            results (dict): Output from search_chunks()
        Returns:
            pd.DataFrame: Flattened and formatted result table
        """
        if not results or "metadatas" not in results:
            print("No results to format.")
            return pd.DataFrame()

        flat_rows = []
        for i in range(len(results["metadatas"][0])):
            row = results["metadatas"][0][i].copy()
            row["Chunk ID"] = results["ids"][0][i]
            row["Distance"] = results["distances"][0][i]
            row["Chunk Text"] = results["documents"][0][i]
            flat_rows.append(row)

        df = pd.DataFrame(flat_rows)
        df = df.sort_values("Distance")
        return df
    
    def run_batch_queries_grouped(self, query_product_pairs, n_results=3):
        """
        Run multiple semantic search queries and group results under business product labels.

        Args:
            query_product_pairs (list of tuples): Each tuple is (query, dataset_product, business_product).
            n_results (int): Number of top results to return per query.

        Returns:
            dict: Mapping from (query, business_product) to formatted DataFrame of results.
        """
        results = {}

        for query, dataset_product, business_product in query_product_pairs:
            print(f"\nQuery: '{query}' | Dataset Product: '{dataset_product}' | Business Product: '{business_product}'")
            raw = self.search_chunks(query=query, n_results=n_results, where={"Product": dataset_product})
            formatted = self.format_search_results(raw)
            display(formatted[["Chunk ID", "Distance", "Company", "Issue", "Chunk Text"]].head(n_results))
            results[(query, business_product)] = formatted

        return results
