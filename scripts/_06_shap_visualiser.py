# _07_shap_visualiser.py
import os
import warnings

import matplotlib.pyplot as plt
import shap
from transformers import AutoTokenizer, pipeline

warnings.simplefilter("ignore", category=FutureWarning)


class SHAPVisualiser:
    def __init__(self, model_path, plot_dir=None):
        """Initialises the SHAPVisualiser class.

        Args:
            model_path (str): Path to fine tuned model.
            complaint (str, optional): Customer complaints summary to visualise.
            plot_dir (str, optional): Directory to save plots.
            Defaults to "Customer reported unauthorized transaction and delayed refund."
        """
        self.model_path = model_path
        self.plot_dir = plot_dir

        # Create directories if they do not exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        print("\n🚀 Initialising SHAPVisualiser class ...")

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

    def load_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.sep_token or "[PAD]"
            )

        summarizer = pipeline(
            "summarization", model=self.model_path, tokenizer=self.tokenizer
        )

        masker = shap.maskers.Text(tokenizer=self.tokenizer)
        self.explainer = shap.Explainer(summarizer, masker)

    def vis(self, complaint):

        print("\n📊 Generating SHAP visualisation...")

        # Compute SHAP values for the input complaint
        shap_values = self.explainer([complaint])
        print("SHAP values type:", type(shap_values))
        print("SHAP values shape:", getattr(shap_values, "shape", None))

        # Visualize token-level impact
        fig = shap.plots.text(shap_values[0])  # This renders the plot

        # Save PNG if plot_dir is defined
        if self.plot_dir:
            plot_path = f"{self.plot_dir}/shap_visualisation.png"
            fig = plt.gcf()  # Get the current figure that was just rendered
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"💾 SHAP text plot saved to {self.safe_relpath(plot_path)}")

        plt.show()
        plt.close()

    def run_shap(
        self, complaint="Customer reported unauthorized transaction and delayed refund."
    ):
        print("\n🧪 Running SHAP visualisation pipeline...")
        self.load_pipeline()
        self.vis(complaint)
