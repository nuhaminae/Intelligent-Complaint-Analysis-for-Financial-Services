import pandas as pd
import os
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class EDA:
    def __init__(self, df_path, plot_dir = None, df_dir = None):
        """
        Initiate EDA class from DataFrame path.

        Args:
            df_path (str): The path to the DataFrame file (e.g., CSV).
            plot_dir (str, optional): The directory to save plots. Defaults to None.
            df_dir (str, optional): The directory to save processed DataFrames. Defaults to None.
        """

        self.df_path = df_path
        self.plot_dir = plot_dir
        self.df_dir = df_dir
        if self.df_path:
            self.load_df()
    
    def load_df (self):
        """
        Load DataFrame and understand the structure of the dataset
            - Number of rows, columns, and data types 
        """

        # Calculate DataFrame relative path 
        rel_df_path = os.path.relpath(self.df_path, os.getcwd())

        if self.df_path:
            try:
                #self.df_raw = pd.read_csv(self.df_path)
                self.df_raw = pd.read_csv(self.df_path, 
                                        dtype={"Consumer complaint narrative": str}, 
                                        low_memory=False)
                if 'Date received' in self.df_raw.columns:
                    self.df_raw['Date received'] = pd.to_datetime(self.df_raw['Date received'], 
                                                                        errors='coerce')
                if 'Date sent to company' in self.df_raw.columns:
                    self.df_raw['Date sent to company'] = pd.to_datetime(self.df_raw['Date sent to company'], 
                                                                        errors='coerce')
                # Capitalise column names for consistency
                self.df_raw.columns = [col.title() for col in self.df_raw.columns]

                print(f"DataFrame loaded successfully from {rel_df_path}")
                print("\nDataFrame head:")
                display (self.df_raw.head())

                print("\nDataFrame tail:")
                display (self.df_raw.tail())
                
                print("\nDataFrame shape:")
                display(self.df_raw.shape)
                
                print("\nDataFrame columns:")
                display(self.df_raw.columns)
                
                print("\nDataFrame summary:")
                self.df_raw.info()
            
            except Exception as e:
                print(f"Error loading DataFrame from {rel_df_path}: {e}")
                self.df_raw = None
        
        else:
            print(f"Error: File not found at {rel_df_path}")
            self.df_raw = None

        return self.df_raw
    
    def save_plot(self, plot_name):
        """
        Saves the current plot to the designated plot folder.
            
        Args:
            plot_name (str): The name of the plot file (including extension, e.g., '.png').
        """
        #create the directory if it doesn't exist
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        plot_path = os.path.join(self.plot_dir, plot_name)
            
        #calculate the relative path
        relative_plot_path = os.path.relpath(plot_path, os.getcwd())
            
        try:
            plt.savefig(plot_path)
            print(f'\nPlot saved to {relative_plot_path}')
        except Exception as e:
            print(f'\nError saving plot: {e}')
    
    @staticmethod
    def clean_narrative(text):
        if pd.isna(text):
            return ""
        
        # Lowercase
        text = text.lower()

        # Expand contractions 
        text = contractions.fix(text)
        
        # Remove boilerplate phrases
        boilerplate_patterns = [
            r"^i am writing to (file|submit|lodge) a complaint.*?",
            r"^to whom it may concern[:,]?",
            r"^hello[:,]?",
            r"^hi[:,]?",
            r"^dear( [a-z]+)?[:,]?",
            r"^this is regarding.*?",
            r"^i am writing to dispute.*?",
            r"^i would like to report.*?",
            r"^i am reaching out.*?",
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove special characters (keep alphanumerics and basic punctuation)
        text = re.sub(r"[^a-z0-9\s.,!?$%&@#-]", "", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        # Normalize unicode
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
        
        # Remove stopwords and lemmatize
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return " ".join(tokens)

#-----------------------------Initial EDA-----------------------------#
    def visualise_complaint(self):
        """
        Visualise the distribution of complaints by product.
        """
        if not hasattr(self, 'df_raw') or self.df_raw is None:
            print("DataFrame not loaded. Please check initialisation.")
            return None
        
        # Count complaints by product
        product_counts = self.df_raw['Product'].value_counts()
        print("Complaint count by product:")
        display(product_counts)
        
        # Plot the distribution of complaints by product
        plt.figure(figsize=(10, 6))
        sns.barplot(y=product_counts.index, x=product_counts.values, 
                    hue=product_counts.index)
        plt.title("Distribution of Complaints by Product")
        plt.xlabel("Number of Complaints")
        plt.ylabel("Product")
        plt.grid()
        plt.tight_layout()

        #select plot directory and plot name to save plot
        plot_name = f"Distribution of Complaints by Product.png"
        self.save_plot (plot_name)
        
        #show plot
        plt.show()
        #close plot to free up space
        plt.close()

    def visualise_complaint_length(self):
        """
        Visualise the length of complaints in the 'Consumer Complaint Narrative' column.
        """
        if not hasattr(self, 'df_raw') or self.df_raw is None:
            print("DataFrame not loaded. Please check initialisation.")
            return None
        
        # Calculate the length of each complaint
        self.df_raw['Complaint Length'] = self.df_raw['Consumer Complaint Narrative'].str.len()
        
        # Plot the distribution of complaint lengths
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df_raw['Complaint Length'], bins=50, kde=True, color='blue')
        plt.title("Distribution of Complaint Lengths")
        plt.xlabel("Length of Complaint")
        plt.ylabel("Frequency")
        plt.grid()
        plt.tight_layout()

        #select plot directory and plot name to save plot
        plot_name = f"Distribution of Complaint Lengths.png"
        self.save_plot (plot_name)
        
        #show plot
        plt.show()
        #close plot to free up space
        plt.close()

    def complaints_narative(self):
        """
        Visualise the 'Consumer Complaint Narrative' column.
        """
        if not hasattr(self, 'df_raw') or self.df_raw is None:
            print("DataFrame not loaded. Please check initialisation.")
            return None
        
        # Total number of complaints
        total_complaints = len(self.df_raw)

        # Complaints with narratives
        with_narrative = self.df_raw['Consumer Complaint Narrative'].notna().sum()

        # Complaints without narratives
        without_narrative = self.df_raw['Consumer Complaint Narrative'].isna().sum()

        print(f"Total complaints: {total_complaints}")
        print(f"Complaints with narratives: {with_narrative}")
        print(f"Complaints without narratives: {without_narrative}")

        #Visualise the distribution of complaints with and without narratives
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["With Narrative", "Without Narrative"], 
                    y=[with_narrative, without_narrative], 
                    hue=["With Narrative", "Without Narrative"])
        plt.title("Complaints With vs. Without Narratives")
        plt.ylabel("Number of Complaints")
        plt.grid()
        plt.tight_layout()
        
        #select plot directory and plot name to save plot
        plot_name = f"Distribution of Complaints With vs Without Narratives.png"
        self.save_plot (plot_name)
        
        #show plot
        plt.show()
        #close plot to free up space
        plt.close()

#-----------------------------Filter Dataset-----------------------------#
    def filter_products(self):
        """
        Filter DataFrame to include only relevant products for analysis.
        """
        if hasattr(self, 'df_raw') and self.df_raw is not None:
            # Define the products of interest
            products_of_interest = [
                'Credit card', 'Credit card or prepaid card',               #credit card
                'Payday loan, little loan, or personal loan',               #personal loan
                'Payday loan, title loan, personal loan, or advance loan',  #personal loan
                'Consumer loan',                                            #personal loan
                'Money transfer',                                           #money transfer
                'Money transfer, virtual currency, or money service',       #money transfer
                'Checking or savings account',                              #savings account
                'Other financial service'                                   #other financial service
            ]
            
            # Filter the DataFrame
            self.df = self.df_raw[self.df_raw['Product'].isin(products_of_interest)].copy()            
            print(f"DataFrame is filtered.")

        else:
            print("DataFrame not loaded. Please check initialisation.")
            return None

    def missing_values(self):
        """
        Identify missing values and handle them appropriately.
        """
        if hasattr(self, 'df') and self.df is not None:
            if self.df.isna().sum().sum() == 0:
                print ('There are no columns with missing values')

            else:
                # Handle missing values in 'Consumer Complaint Narrative'
                if self.df['Consumer Complaint Narrative'].isna().sum() > 0:
                    self.df.dropna(subset=['Consumer Complaint Narrative'], inplace=True)
                    print('\nRows with missing "Consumer Complaint Narrative" have been dropped.')
                
                # Identify and fill the rest of the missing values with "Unknown"
                mis_values = self.df.isna().sum()
                missing_cols = mis_values[mis_values > 0].index.tolist()
                self.df[missing_cols] = self.df[missing_cols].fillna("Unknown")
                print('\nColumns with with missing values are filled with "Unknown":', missing_cols)

                self.df = self.df.reset_index(drop=True)
        else:
            print("DataFrame not loaded. Please check initialisation.")
            return None
        
#-----------------------------Normalise Dataset Text-----------------------------#
    def normalise_text(self):
        """
        Normalise text under 'Consumer Complaint Narrative' columns
        """
        if not hasattr(self, 'df') or self.df is None:
            print("\nNo processed DataFrame found. Please run preprocessing steps before saving.")
            return
        
        # Change  column to lower case
        self.df['Clean Consumer Complaint Narrative'] = self.df['Consumer Complaint Narrative'
                                                                ].apply(self.clean_narrative)
        print ("\nText under 'Consumer Complaint Narrative' column are normalised.")

#--------------------------------------------------------------------------------#
    def save_df(self, filename = 'filtered_complaints.csv'):
        """
        Saves the processed DataFrame to the specified directory.

        Args:
            filename (str, optional): The name of the output file. Defaults to 'filtered_complaints.csv'.
        """

        if not hasattr(self, 'df') or self.df is None:
            print("\nNo processed DataFrame found. Please run preprocessing steps before saving.")
            return

        # Create output folder if it doesn't exist
        if not os.path.exists(self.df_dir):
            os.makedirs(self.df_dir)
            
        df_name = os.path.join(self.df_dir, filename)
            
        # Calculate the relative path
        relative_path = os.path.relpath(df_name, os.getcwd())
            
        # Sort and save processed DataFrame to CSV
        self.df = self.df[sorted(self.df.columns)]
        self.df.to_csv(df_name, index=False)
        print(f'\nProcessed DataFrame saved to: {relative_path}')

        print('\nDataFrame Head:')
        display (self.df.head())
            
        print(f"\nProcessed DataFrame shape: {self.df.shape}")
