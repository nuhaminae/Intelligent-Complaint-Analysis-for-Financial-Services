from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from difflib import get_close_matches
from tqdm import tqdm
import pandas as pd
from langchain_openai import ChatOpenAI
from difflib import SequenceMatcher
import re
import os

# To Load OpenAI key
from dotenv import load_dotenv
load_dotenv() 
openai_api_key = os.getenv('OPENAI_API_KEY')

class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline for answering questions about customer complaints
    across financial products using ChromaDB and Hugging Face models.
    """

    def __init__(self, chroma_client, collection_name="complaints_chunks",
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                llm_model_name="google/flan-t5-base", eval_dir=None):
        """
        Initialise the RAG pipeline with ChromaDB and Hugging Face models.

        Args:
            chroma_client: Existing ChromaDB PersistentClient.
            collection_name (str): Name of the ChromaDB collection.
            embedding_model_name (str): SentenceTransformer model for embeddings.
            llm_model_name (str): Hugging Face model for generation.
            eval_dir (str): Directory where evaluation CSV/Markdown files will be saved.
        """
        print("Initialising RAGPipeline...")
        self.eval_dir = eval_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.retriever = Chroma(client=chroma_client, collection_name=collection_name,
                                embedding_function=self.embedding_model).as_retriever(search_kwargs={"k": 5})

        self.prompt_template = PromptTemplate.from_template(
    """You are a financial analyst assistant for CrediTrust. Your task is to generate clear, insightful summaries of customer complaints.

Use the retrieved excerpts to answer the question. Write in full sentences using formal, grammatically correct language. Do not copy excerpts directly. Instead, extract key themes and explain them concisely in your own words. Avoid repetition. If the context lacks a clear answer, say so politely.

Context:
{context}

Question:
{question}

Answer:""")


        self.llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model=llm_model_name, max_length=256))
        #self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

        '''
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever,
                                                    chain_type="stuff",
                                                    chain_type_kwargs={"prompt": self.prompt_template},
                                                    return_source_documents=True)
        '''
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever,
                                                    chain_type="refine", chain_type_kwargs={
                                                        "question_prompt": self.prompt_template,
                                                        "refine_prompt": self.prompt_template,
                                                        "document_variable_name": "context"},
                                                    return_source_documents=True)

        print("RAGPipeline initialised.")

    def safe_relpath(self, path, start=os.getcwd()):
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
    
    def clean_chunks(self, chunks, min_words=20, max_words=150, max_chunks=3, similarity_threshold=0.9):
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

        def ends_in_complete_sentence(text):
            # Ensures chunk ends with a sentence delimiter
            return bool(re.match(r".+[.!?]$", text.strip()))

        seen = []
        cleaned = []

        for chunk in chunks:
            text = chunk.page_content.strip()
            word_count = len(text.split())

            # Skip noisy or short chunks
            if word_count < min_words or is_profane_or_noisy(text) or not ends_in_complete_sentence(text):
                continue

            # Truncate long chunks cleanly
            if word_count > max_words:
                text = " ".join(text.split()[:max_words])
                text = text.strip(" .!?,;:")
                chunk.page_content = text

            # Remove near-duplicates using SequenceMatcher
            if any(SequenceMatcher(None, text, prev).ratio() > similarity_threshold for prev in seen):
                continue

            seen.append(text)
            cleaned.append(chunk)

            if len(cleaned) >= max_chunks:
                break

        return cleaned

    def run(self, question):
        """
        Run the full RAG pipeline for a given question.

        Args:
            question (str): User query.
        Returns:
            dict: Contains question, answer, and source documents.
        """
        print("-" * 100)
        print(f"Running RAG pipeline for: {question}")

        # Retrieve and clean chunks
        retrieved_docs = self.retriever.invoke(question)
        cleaned_docs = self.clean_chunks(retrieved_docs)

        # Generate answer
        if not cleaned_docs:
            answer = "I don't have enough information to answer this question based on the available documents."
        else:
            result = self.qa_chain.combine_documents_chain.invoke({
                "input_documents": cleaned_docs,
                "question": question
            })
            def polish_answer(text):
                # Remove leading punctuation
                text = re.sub(r"^[^\w]+", "", text)
                # Capitalize first letter
                text = text[0].upper() + text[1:] if text else text
                # Trim excessive whitespace
                return text.strip()

            #answer = result["output_text"] if isinstance(result, dict) and "output_text" in result else result
            answer = polish_answer(result["output_text"]) if isinstance(result, dict) else polish_answer(result)

        print(f"\nAnswer:\n{answer}")
        print(f"\nSources retrieved: {len(cleaned_docs)}")
        print("-" * 100)

        return {
            "question": question,
            "answer": answer,
            "sources": [doc.page_content for doc in cleaned_docs]
        }

    def evaluate(self, questions, filename="rag_eval.csv"):
        """
        Run qualitative evaluation on a list of questions and save results to CSV.

        Args:
            questions (list): List of user questions to evaluate.
            filename (str): Output CSV filename.

        Returns:
            pd.DataFrame: Evaluation results.
        """
        print(f"Starting evaluation on {len(questions)} questions...")
        records = []
        for q in tqdm(questions, desc="Evaluating"):
            result = self.run(q)
            records.append({
                "Question": result["question"],
                "Generated Answer": result["answer"],
                "Retrieved Source 1": result["sources"][0] if result["sources"] else "",
                "Retrieved Source 2": result["sources"][1] if len(result["sources"]) > 1 else "",
                "Quality Score (1-5)": "",
                "Comments": ""
            })

        eval_df = pd.DataFrame(records)
        if self.eval_dir:
            os.makedirs(self.eval_dir, exist_ok=True)
            output_path = os.path.join(self.eval_dir, filename)
            eval_df.to_csv(output_path, index=False)
            print(f"\nEvaluation complete. Results saved to: {self.safe_relpath(output_path)}")
        else:
            print("\nEvaluation directory not specified. Skipping CSV export.")
        return eval_df

    def generate_markdown_table(self, df, filename="rag_evaluation.md"):
        """
        Convert evaluation DataFrame to a Markdown table with highlights for low scores and hallucinations.
        """
        def highlight(score, comment):
            try:
                score = int(score)
            except:
                return ""
            if score <= 2:
                return "🚨"
            elif "hallucination" in comment.lower():
                return "⚠️"
            return ""

        if self.eval_dir:
            os.makedirs(self.eval_dir, exist_ok=True)
            output_path = os.path.join(self.eval_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("##RAG System Evaluation\n\n")
                f.write("| Question | Generated Answer | Retrieved Source 1 | Retrieved Source 2 | Quality Score (1–5) | Comments |\n")
                f.write("|----------|------------------|---------------------|---------------------|----------------------|----------|\n")
                for _, row in df.iterrows():
                    flag = highlight(row["Quality Score (1-5)"], row["Comments"])
                    f.write(f"| {row['Question']} | {row['Generated Answer']} | {row['Retrieved Source 1']} | {row['Retrieved Source 2']} | {row['Quality Score (1-5)']} {flag} | {row['Comments']} |\n")
            print(f"\nMarkdown evaluation table saved to: {self.safe_relpath(output_path)}")
        else:
            print("\nEvaluation directory not specified. Skipping Markdown export.")