# _04_rag_core_logic.py

import json
import os
import re
from difflib import SequenceMatcher

import pandas as pd
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# from difflib import get_close_matches
from tqdm import tqdm
from transformers import pipeline

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline for answering questions about customer complaints
    across financial products using ChromaDB and Hugging Face models.
    """

    def __init__(
        self,
        chroma_client,
        collection_name="complaints_chunks",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="google/flan-t5-base",
        eval_dir=None,
    ):
        """
        Initialise the RAG pipeline with ChromaDB and Hugging Face models.

        Args:
            chroma_client: Existing ChromaDB PersistentClient.
            collection_name (str): Name of the ChromaDB collection.
            embedding_model_name (str): SentenceTransformer model for embeddings.
            llm_model_name (str): Hugging Face model for generation.
            eval_dir (str): Directory where evaluation CSV/Markdown files will be saved.
        """
        self.eval_dir = eval_dir

        self.conversation_history = []
        self.max_history_turns = 3

        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        print("🚀 Initialising RAGPipeline...\n")

        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.retriever = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        ).as_retriever(search_kwargs={"k": 10})

        self.prompt_template = PromptTemplate.from_template(
            """You are a financial analyst assistant for CrediTrust. Your task is to generate clear, insightful summaries of customer complaints across financial products.

        Instructions:
        - First, analyse the question carefully to understand what financial issue is being asked.
        - Then, synthesise key insights from the retrieved complaint excerpts.
        - Write in full sentences using formal, grammatically correct language.
        - Do not copy excerpts directly. Instead, extract themes and explain them concisely in your own words.
        - Avoid repetition and vague statements.
        - If the context lacks a clear answer, say so politely and suggest what additional data might help.

        Previous exchanges:
        {history}

        Context:
        {context}

        Current Question:
        {question}

        Answer:"""
        )

        self.llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text2text-generation", model=llm_model_name, max_length=256
            )
        )
        # self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

        """
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever,
                                                    chain_type="stuff",
                                                    chain_type_kwargs={"prompt": self.prompt_template},
                                                    return_source_documents=True)
        """
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="refine",
            chain_type_kwargs={
                "question_prompt": self.prompt_template,
                "refine_prompt": self.prompt_template,
                "document_variable_name": "context",
            },
            return_source_documents=True,
        )

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

    def format_history(self):
        """
        Format the conversation history for prompt injection.
        """
        history_str = ""
        for turn in self.conversation_history[-self.max_history_turns :]:
            history_str += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
        return history_str.strip()

    def reset_history(self):
        """
        Reset the conversation history.
        """
        self.conversation_history = []

    def clean_chunks(
        self,
        chunks,
        min_words=30,
        max_words=120,
        max_chunks=3,
        similarity_threshold=0.95,
    ):
        """
        Clean and filter retrieved chunks to improve LLM answer quality.

        Args:
            chunks (list): List of Document objects.
            min_words (int): Minimum word count to keep a chunk.
            max_words (int): Maximum word count to truncate a chunk.
            max_chunks (int): Maximum number of diverse chunks to return.
            similarity_threshold (float): Threshold for filtering near-duplicate chunks.

        Returns:
            list: Cleaned and filtered list of Document objects.
        """

        def is_profane_or_noisy(text):
            # Filters profanity, censored words, and special character overload
            return bool(re.search(r"(xxxx|f\*+|s\*+|[^\w\s]{4,})", text.lower()))

        seen = []
        cleaned = []

        for chunk in chunks:
            text = chunk.page_content.strip()
            word_count = len(text.split())

            # Skip noisy or short chunks
            if word_count < min_words or is_profane_or_noisy(text):
                continue

            # Truncate long chunks cleanly
            if word_count > max_words:
                text = " ".join(text.split()[:max_words])
                text = text.strip(" .!?,;:")
                chunk.page_content = text

            # Remove near-duplicates using SequenceMatcher
            if any(
                SequenceMatcher(None, text, prev).ratio() > similarity_threshold
                for prev in seen
            ):
                continue

            seen.append(text)
            cleaned.append(chunk)

            if len(cleaned) >= max_chunks:
                break

        return cleaned

    def run_rag(self, question):
        """
        Run the full RAG pipeline for a given question.

        Args:
            question (str): User query.
        Returns:
            dict: Contains question, answer, and source documents.
        """

        print(f"\n❔ Running RAG pipeline for: {question}")

        retrieved_docs = self.retriever.invoke(question)
        cleaned_docs = self.clean_chunks(retrieved_docs)

        if not cleaned_docs:
            answer = "I don't have enough information to answer this question based on the available documents."
        else:
            history_str = self.format_history()
            result = self.qa_chain.combine_documents_chain.invoke(
                {
                    "input_documents": cleaned_docs,
                    "question": question,
                    "history": history_str,
                }
            )

            def polish_answer(text):
                text = re.sub(r"^[^\w]+", "", text)
                return text[0].upper() + text[1:] if text else text.strip()

            answer = (
                polish_answer(result["output_text"])
                if isinstance(result, dict)
                else polish_answer(result)
            )

        # Store the turn
        self.conversation_history.append({"question": question, "answer": answer})

        print(f"\n📌 Answer:\n {answer}")
        print(f"\n📌 Sources retrieved:\n {len(cleaned_docs)}")
        print("🔹 " * 25)

        return {
            "question": question,
            "answer": answer,
            "sources": [doc.page_content for doc in cleaned_docs],
        }

    def evaluate(self, questions, filename="rag_evaluation.csv"):
        """
        Run qualitative evaluation on a list of questions and save results to CSV.

        Args:
            questions (list): List of user questions to evaluate.
            filename (str): Output CSV filename.

        Returns:
            pd.DataFrame: Evaluation results.
        """
        print(f"🧪 Starting evaluation on {len(questions)} questions...")
        records = []
        for q in tqdm(questions, desc="Evaluating"):
            result = self.run_rag(q)
            records.append(
                {
                    "Question": result["question"],
                    "Generated Answer": result["answer"],
                    "Retrieved Source 1": (
                        result["sources"][0] if result["sources"] else ""
                    ),
                    "Retrieved Source 2": (
                        result["sources"][1] if len(result["sources"]) > 1 else ""
                    ),
                    "Retrieved Source 3": (
                        result["sources"][2] if len(result["sources"]) > 2 else ""
                    ),
                    "Quality Score (1-5)": "",
                    "Comments": "",
                }
            )

        self.eval_df = pd.DataFrame(records)
        if self.eval_dir:
            os.makedirs(self.eval_dir, exist_ok=True)
            output_path = os.path.join(self.eval_dir, filename)
            self.eval_df.to_csv(output_path, index=False)
            print(
                f"\n💾 Evaluation complete. Results saved to: {self.safe_relpath(output_path)}"
            )
        else:
            print("\n⚠️ Evaluation directory not specified. Skipping CSV export.")
        return self.eval_df

    def generate_markdown_table(self, df, filename="rag_evaluation.md"):
        """
        Convert evaluation DataFrame to a Markdown table with highlights for low scores and hallucinations.
        Adds chunk length metadata for traceability.
        """

        def highlight(score, comment):
            try:
                score = int(score)
            except Exception:
                score = None

            comment = str(comment).lower() if isinstance(comment, str) else ""

            if score is not None and score <= 2:
                return "🚨"
            elif "hallucination" in comment:
                return "⚠️"
            return ""

        def chunk_length(text):
            return len(text.split()) if isinstance(text, str) else 0

        def format_row(row):
            flag = highlight(row["Quality Score (1-5)"], row["Comments"])
            source1_len = chunk_length(row["Retrieved Source 1"])
            source2_len = chunk_length(row["Retrieved Source 2"])
            source3_len = chunk_length(row["Retrieved Source 3"])
            return (
                f"| {row['Question']} | {row['Generated Answer']} | "
                f"{row['Retrieved Source 1']} | {row['Retrieved Source 2']} | {row['Retrieved Source 3']} | "
                f"{row['Quality Score (1-5)']} {flag} | {row['Comments']} | "
                f"{source1_len}/{source2_len}/{source3_len} |"
            )

        if not self.eval_dir:
            print("\n⚠️ Evaluation directory not specified. Skipping Markdown export.")
            return

        output_path = os.path.join(self.eval_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# RAG System Evaluation\n\n")
            f.write(
                "| Question | Generated Answer | Retrieved Source 1 | Retrieved Source 2 | "
                "Retrieved Source 3 | Quality Score (1–5) | Chunk Lengths (1/2/3)  |Comments |\n"
            )
            f.write(
                "|----------|------------------|---------------------|---------------------|"
                "---------------------|----------------------|----------|--------------------|\n"
            )
            for _, row in df.iterrows():
                f.write(format_row(row) + "\n")

        print(
            f"\n💾 Markdown evaluation table saved to: {self.safe_relpath(output_path)}"
        )

    def to_training_pair(self, row):
        sources = [row.get(f"Retrieved Source {i}", "") for i in range(1, 4)]
        context = "\n".join(
            [src for src in sources if isinstance(src, str) and src.strip()]
        )
        return {
            "instruction": "Summarise the following complaint formally and concisely.",
            "input": f"Question: {row['Question']}\nContext:\n{context}",
            "output": row["Generated Answer"],
        }

    def convert_to_training_pairs(self, df, filename="complaint_response_pairs.jsonl"):
        """Convert a DataFrame of complaints and responses into training pairs."""

        # Ensure the column is numeric
        df["Quality Score (1-5)"] = pd.to_numeric(
            df["Quality Score (1-5)"], errors="coerce"
        )

        # Drop rows with NaN scores
        df = df.dropna(subset=["Quality Score (1-5)"])

        # Filter for high-quality responses
        df = df[df["Quality Score (1-5)"] >= 3]

        # Convert
        pairs = [self.to_training_pair(row) for _, row in df.iterrows()]

        # Save as JSONL
        output_path = os.path.join(self.eval_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        print(
            f"💾 Saved {len(pairs)} training pairs to {self.safe_relpath(output_path)}"
        )
