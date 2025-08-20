# app.py

import time

import gradio as gr
from chromadb import PersistentClient

from scripts._04_rag_core_logic import RAGPipeline

# Load vector store and pipeline
chroma_client = PersistentClient(path="vector store/")
rag = RAGPipeline(
    chroma_client=chroma_client,
    eval_dir="data/evaluation",
    llm_model_name="model_finetuned_flant5_v1",
)

product_options = [
    "All",
    "Credit card",
    "Payday Loan/BNPL",
    "Personal loan",
    "Savings account",
    "Money transfers",
]
issue_options = [
    "All",
    "Billing dispute",
    "Delayed response",
    "Unauthorized charges",
    "Account closure",
    "Fraud",
    "Other",
]


def ask(question, product, issue):
    filters = {}
    if product != "All":
        filters["Product"] = product
    if issue != "All":
        filters["Issue"] = issue

    start_time = time.time()
    result = rag.run_rag(question)
    end_time = time.time()

    elapsed = round(end_time - start_time, 2)
    answer = result["answer"]
    sources = result["sources"]
    sources_display = "\n\n".join([f"🔹 {src}" for src in sources])
    elapsed_display = f"⏱️ Retrieved in {elapsed} seconds"

    return answer, sources_display, elapsed_display


def clear():
    rag.reset_history()
    return "", ""


with gr.Blocks() as demo:
    # Header
    gr.Markdown(
        """
    <div style='background-color: #FDFEFE; padding: 20px; border-radius: 10px;'>
        <h1 style='color: #1A237E;
        text-align: center;'>CrediTrust Complaint Analyst</h1>
        <p style='color: #455A64; font-size: 18px; text-align: center;'>
            A transparent NLP solution for surfacing high-risk financial complaints.
        </p>
    </div>
    """
    )

    # Action Banner
    gr.Markdown(
        """
    <div style='background-color: #F9A825; padding: 10px;
    border-radius: 8px; margin-top: 20px;'>
        <p style='color: white; font-weight: bold; text-align: center;'>
            🖱️ Click 'Ask' to begin. Use filters to refine by product or issue. \
            Click 'Clear History' to reset context.
        </p>
    </div>
    """
    )

    # Question Input
    question_input = gr.Textbox(lines=1, placeholder="Ask about customer complaints...")

    # Buttons
    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear History", variant="secondary")

    # Filters
    with gr.Group():
        gr.Markdown("<h3 style='color: #1A237E;'>🔍 Complaint Filters</h3>")
        with gr.Row():
            product_dropdown = gr.Dropdown(
                choices=product_options, label="Product Filter", value="All"
            )
            issue_dropdown = gr.Dropdown(
                choices=issue_options, label="Issue Filter", value="All"
            )

    # Outputs
    with gr.Column():

        gr.Markdown(
            "<div style='background-color: #E8F5E9;"
            "padding: 10px; border-radius: 8px;'>"
            "<h3 style='color: #2E7D32;'>Answer</h3></div>"
        )
        answer_output = gr.Textbox(lines=3, label="", show_copy_button=True)
        elapsed_time_display = gr.Textbox(label="Time Elapsed", interactive=False)

        gr.Markdown(
            "<div style='background-color: #FFF3E0;"
            "padding: 10px; border-radius: 8px;'>"
            "<h3 style='color: #F9A825;'>Retrieved Sources</h3></div>"
        )
        sources_output = gr.Textbox(lines=3, label="", show_copy_button=True)

    # Footer
    gr.Markdown(
        """
    <div style='margin-top: 30px; text-align: center; font-size: 14px; color: #78909C;'>
        Built with 💡 by Nuhamin | Model: finetuned Flan-T5 | Last updated: August 2025
    </div>
    """
    )

    # Button Logic
    submit_btn.click(
        fn=ask,
        inputs=[question_input, product_dropdown, issue_dropdown],
        outputs=[answer_output, sources_output, elapsed_time_display],
    )

    clear_btn.click(
        fn=clear, outputs=[answer_output, sources_output, elapsed_time_display]
    )

if __name__ == "__main__":
    demo.launch()
    # demo.launch(share=True)
