# app.py

import os
import sys
import gradio as gr
from chromadb import PersistentClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts._04_rag_core_logic import RAGPipeline

# Load vector store and pipeline
chroma_client = PersistentClient(path="vector store/")
rag = RAGPipeline(
    chroma_client=chroma_client, 
    eval_dir="data/evaluation", 
    llm_model_name="model"
    )


product_options = ["All", "Credit card", "Payday Loan/BNPL", "Personal loan", "Savings account", "Money transfers"]
issue_options = ["All", "Billing dispute", "Delayed response", "Unauthorized charges", "Account closure"]

def ask(question, product, issue):
    filters = {}
    if product != "All":
        filters["Product"] = product
    if issue != "All":
        filters["Issue"] = issue

    result = rag.run(question, filters=filters)
    answer = result["answer"]
    sources = result["sources"]
    sources_display = "\n\n".join([f"🔹 {src}" for src in sources])
    return answer, sources_display

def clear():
    rag.reset_history()
    return "", ""

with gr.Blocks() as demo:
    gr.Markdown("## CrediTrust Complaint Analyst 🤖")
    gr.Markdown("Ask questions about customer complaints. \
                \
                Filter by product or issue. Click 'Clear' to reset context.")

    with gr.Row():
        question_input = gr.Textbox(lines=2, placeholder="Ask about customer complaints...")
        clear_btn = gr.Button("Clear History")

    with gr.Row():
        product_dropdown = gr.Dropdown(choices=product_options, label="Product Filter", value="All")
        issue_dropdown = gr.Dropdown(choices=issue_options, label="Issue Filter", value="All")

    answer_output = gr.Textbox(label="Answer", lines=5)
    sources_output = gr.Textbox(label="Retrieved Sources", lines=8)

    submit_btn = gr.Button("Ask")

    submit_btn.click(fn=ask, inputs=[question_input, product_dropdown, issue_dropdown],
                    outputs=[answer_output, sources_output])
    clear_btn.click(fn=clear, outputs=[answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()
