import os
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json

class LangchainIndexer:
    def __init__(self, df_chunks_path, vector_store_dir="vector_z/", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the LangchainIndexer using LangChain's Chroma wrapper.

        Args:
            df_chunks_path (str): Path to the chunked DataFrame file (CSV).
            vector_store_dir (str): Directory to persist the ChromaDB vector store.
            model_name (str): SentenceTransformer model name.
        """
        self.df_chunks_path = df_chunks_path
        self.vector_store_dir = vector_store_dir
        self.model_name = model_name
        self.df_chunks = None

        os.makedirs(self.vector_store_dir, exist_ok=True)

        self.embedding_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")
        self.vectorstore = Chroma(
            collection_name="complaints_chunks",
            embedding_function=self.embedding_model,
            persist_directory=self.vector_store_dir
        )

        print(f"Using embedding model: {self.model_name}")

    def load_chunks(self):
        """
        Load the chunked DataFrame from CSV.
        """
        try:
            self.df_chunks = pd.read_csv(self.df_chunks_path)
            if "Chunk" not in self.df_chunks.columns:
                raise ValueError("Missing 'Chunk' column in the DataFrame.")
            print(f"\nLoaded {len(self.df_chunks)} chunks from {self.df_chunks_path}")
        except Exception as e:
            print(f"\nError loading DataFrame: {e}")
            self.df_chunks = None


    def index_chunks(self, batch_size=5000, resume_from=0, failed_log_path="failed_batches.json"):
        """
        Convert DataFrame rows into LangChain Documents and index them in Chroma in batches.

        Args:
            batch_size (int): Number of documents to index per batch.
            resume_from (int): Batch index to resume from (0-based).
            failed_log_path (str): Path to save failed batch indices for retry.
        """
        if self.df_chunks is None:
            raise ValueError("DataFrame not loaded. Run load_chunks() first.")

        existing = self.vectorstore._collection.count()
        if existing > 0 and resume_from == 0:
            print(f"\nSkipping indexing: {existing} documents already exist.")
            return

        print(f"\nIndexing {len(self.df_chunks)} complaint chunks into ChromaDB in batches of {batch_size}...\n")

        documents = []
        for _, row in tqdm(self.df_chunks.iterrows(), total=len(self.df_chunks), desc="Preparing documents"):
            metadata = row.to_dict()
            content = metadata.pop("Chunk", "").strip()
            if not content:
                continue
            documents.append(Document(page_content=content, metadata=metadata))

        failed_batches = []

        for i in range(resume_from * batch_size, len(documents), batch_size):
            batch_num = i // batch_size
            batch = documents[i:i + batch_size]
            try:
                self.vectorstore.add_documents(batch)
                print(f"\nIndexed batch {batch_num + 1} ({len(batch)} documents)")
            except Exception as e:
                print(f"\nFailed to index batch {batch_num + 1}: {e}")
                failed_batches.append(batch_num)

        #self.vectorstore.persist()
        print(f"\nVector store saved to: {self.vector_store_dir}\n")

        if failed_batches:
            with open(failed_log_path, "w") as f:
                json.dump(failed_batches, f)
            print(f"\nFailed batches logged to: {failed_log_path}")
        else:
            print("All batches indexed successfully.")

    def search(self, query, k=3, filter=None):
        """
        Perform semantic search using LangChain's retriever.

        Args:
            query (str): Search query.
            k (int): Number of top results.
            filter (dict): Optional metadata filter.

        Returns:
            list: Retrieved Document objects.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k, "filter": filter})
        return retriever.invoke(query)

    def preview_results(self, query, k=3, filter=None):
        """
        Display top-k search results for a query.
        """
        results = self.search(query, k=k, filter=filter)
        # Print the top result
        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Product: {doc.metadata.get('Product')}")
            print(f"Issue: {doc.metadata.get('Issue')}")
            print(f"Chunk:{doc.page_content}\n")