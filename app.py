import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from scripts._03_rag_core_logic import RAGPipeline
from chromadb import PersistentClient

# Load vector store and pipeline
chroma_client = PersistentClient(path="vector store/")
rag = RAGPipeline(chroma_client=chroma_client, eval_dir="data/evaluation")

def ask(question):
    result = rag.run(question)
    answer = result["answer"]
    sources = result["sources"]
    sources_display = "\n\n".join([f"🔹 {src}" for src in sources])
    return answer, sources_display

# Interface
demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=2, placeholder="Ask about customer complaints..."),
    outputs=[
        gr.Textbox(label="Answer", lines=5),
        gr.Textbox(label="Retrieved Sources", lines=8)
    ],
    title="CrediTrust Complaint Analyst 🤖",
    description="Ask any question about customer complaints across products like BNPL, loans, savings, money stransfer etc. The system retrieves and summarises relevant feedback.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()