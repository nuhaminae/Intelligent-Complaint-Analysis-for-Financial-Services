import pandas as pd
import os
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunking:
    def __init__(self, df_path, plot_dir = None, df_dir = None):
        """
        Initiate Chunking class from DataFrame path.

        Args:
            df_path (str): The path to the DataFrame file (e.g., CSV).
            plot_dir (str, optional): The directory to save plots. Defaults to None.
            df_dir (str, optional): The directory to save processed DataFrames. Defaults to None.
        """

        self.df_path = df_path
        self.plot_dir = plot_dir
        self.df_dir = df_dir
        
        # Initialise the splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,                            # Start with 500 characters
            chunk_overlap=100,                         # 100-character overlap
            separators=["\n\n", "\n", ".", " ", ""])
        
        if self.df_path:
            self.load_df()
    
    @staticmethod
    def safe_relpath(path, start=os.getcwd()):
        try:
            return os.path.relpath(path, start)
        except ValueError:
            return path  # fallback to absolute path if on different drives
        
    def load_df (self):
        """
        Load DataFrame        
        """

        # Calculate DataFrame relative path 
        rel_df_path = self.safe_relpath(self.df_path)

        if self.df_path:
            try:
                self.df = pd.read_csv(self.df_path)
                if 'Date Received' in self.df.columns:
                    self.df['Date Received'] = pd.to_datetime(self.df['Date Received'], 
                                                                        errors='coerce')
                if 'Date Sent To Company' in self.df.columns:
                    self.df['Date Sent To Company'] = pd.to_datetime(self.df['Date Sent To Company'], 
                                                                        errors='coerce')

                # Rename and drop if applicable
                if 'Clean Consumer Complaint Narrative' in self.df.columns:
                    self.df['Consumer Complaint Narrative Clean'] = self.df['Clean Consumer Complaint Narrative']
                    self.df.drop(columns=['Clean Consumer Complaint Narrative'], inplace=True)
                    print("Renamed 'Clean Consumer Complaint Narrative' to 'Consumer Complaint Narrative Clean' and dropped the original column.")

                print(f"\nDataFrame loaded successfully from {rel_df_path}")

            except Exception as e:
                print(f"Error loading DataFrame from {rel_df_path}: {e}")
                self.df = None
        
        else:
            print(f"Error: File not found at {rel_df_path}")
            self.df = None

        return self.df
    
#-----------------------------Chunking-----------------------------#
    def chunk_narrative(self):
        """
        Chunk narratives using RecursiveCharacterTextSplitter and explode into multiple rows.
        Saves the chunked DataFrame to self.df_chunks.
        """
        
        if hasattr(self, 'df') and self.df is not None:
            if "Consumer Complaint Narrative Clean" not in self.df.columns:
                print("\nError: 'Consumer Complaint Narrative Clean' column not found in DataFrame.")
                return None
            
            # Apply chunking
            self.df["Chunk"] = self.df["Consumer Complaint Narrative Clean"].apply(
                lambda x: self.text_splitter.split_text(str(x)) if pd.notnull(x) else []
            )
            
            # Explode into multiple rows
            self.df_chunks = self.df.explode("Chunk").reset_index(drop=True)
            
            print("Chunking complete.")
            
            return self.df_chunks
        
        else:
            print("\nNo DataFrame found. Please ensure the DataFrame is loaded correctly.")
            return None

    def plot_chunk_lengths(self):
        """
        Plot distribution of chunk lengths.
        """
        if hasattr(self, 'df_chunks') and self.df_chunks is not None:
            self.df_chunks["Chunk Length"] = self.df_chunks["Chunk"].apply(len)
            
            # Sort Columns
            self.df_chunks = self.df_chunks[sorted(self.df_chunks.columns)]
            
            print("\nChunked DataFrame head:")
            display (self.df_chunks.head())
                
            print(f"\nDataFrame shape after exploding (Chunked DataFrame): {self.df_chunks.shape}")
            print(f"DataFrame shape before exploding: {self.df.shape}")
                
            print("\nChunked DataFrame summary:")
            self.df_chunks.info()

            plt.figure(figsize=(10, 6))
            sns.histplot(self.df_chunks["Chunk Length"], bins = 50, kde = True, color = 'green')
            plt.title("Distribution of Chunk Lengths")
            plt.xlabel("Characters per Chunk")
            plt.ylabel("Frequency")
            plt.grid()
            plt.tight_layout()

            if self.plot_dir:
                plot_path = os.path.join(self.plot_dir, "Distribution of Chunk Lengths.png")
                rel_plot_path = self.safe_relpath(plot_path)
                plt.savefig(plot_path)
                print(f"\nPlot saved to {rel_plot_path}")
            
            #show and close plot
            plt.show()
            fig = plt.gcf()  # Get current figure
            plt.close()
            return fig
        
        else:
            print("No chunked DataFrame found. Please run chunk_narrative() first.")

#--------------------------------------------------------------------------------#
    def save_df(self, filename = 'chunked_complaints.csv'):
        """
        Saves the processed DataFrame to the specified directory.

        Args:
            filename (str, optional): The name of the output file. Defaults to 'chunked_complaints.csv'.
        """

        if not hasattr(self, 'df_chunks') or self.df_chunks is None:
            print("\nNo DataFrame found. Please ensure the DataFrame is loaded correctly")
            return

        # Create output folder if it doesn't exist
        if not os.path.exists(self.df_dir):
            os.makedirs(self.df_dir)
            
        df_name = os.path.join(self.df_dir, filename)

        rel_df_path = self.safe_relpath(df_name)
            
        # Save chunked DataFrame to CSV
        self.df_chunks.to_csv(df_name, index=False)
        print(f'\nChunked DataFrame saved to: {rel_df_path}')

