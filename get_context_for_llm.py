import pandas as pd
import nbformat
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

class DataContextBuilder:
    def __init__(self):
        """Initialize the context builder."""
        self.train_df = None
        self.test_df = None
        self.notebook_content = None
        self.user_responses = {}

    def load_data(self, train_path: str, test_path: str = None) -> None:
        """Load train and optionally test data."""
        try:
            self.train_df = pd.read_csv(train_path)
            if test_path:
                self.test_df = pd.read_csv(test_path)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def load_notebook(self, notebook_path: str) -> None:
        """Load and parse IPython notebook."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                self.notebook_content = nbformat.read(f, as_version=4)
        except Exception as e:
            raise Exception(f"Error loading notebook: {str(e)}")

    def generate_context(self) -> str:
        """Generate context for LLM from all available information."""
        try:
            if self.train_df is None:
                raise ValueError("Training data not loaded")

            context_parts = []

            # Add data analysis information
            context_parts.append("=== TRAINING DATA ANALYSIS ===")
            context_parts.append(self._get_dataframe_info(self.train_df, "Training"))

            if self.test_df is not None:
                context_parts.append("\n=== TEST DATA ANALYSIS ===")
                context_parts.append(self._get_dataframe_info(self.test_df, "Test"))

            # Add notebook analysis if available
            if self.notebook_content is not None:
                context_parts.append("\n=== NOTEBOOK ANALYSIS ===")
                context_parts.append(self._analyze_notebook())

            # Combine all parts
            full_context = "\n".join(context_parts)

            # Use OpenAI to enhance the context
            try:
                enhanced_context = self._enhance_context(full_context)
            except Exception as e:
                print(f"Warning: Could not enhance context with OpenAI: {str(e)}")
                enhanced_context = full_context

            return full_context+enhanced_context

        except Exception as e:
            raise Exception(f"Error generating context: {str(e)}")

    def _get_dataframe_info(self, df: pd.DataFrame, dataset_type: str) -> str:
        """Get comprehensive information about a dataframe."""
        try:
            info_parts = [
                f"{dataset_type} Dataset Shape: {df.shape}",
                f"\nDataframe Info:\n{df.info()}",
                f"\nDescriptive Statistics:\n{df.describe().to_string()}",
                f"\nMissing Values:\n{df.isnull().sum().to_string()}",
                f"\nData Types:\n{df.dtypes.to_string()}"
            ]
            return "\n".join(info_parts)
        except Exception as e:
            return f"Error getting dataframe info: {str(e)}"

    def _analyze_notebook(self) -> str:
        """Analyze notebook content and extract relevant information."""
        try:
            if self.notebook_content is None:
                return "No notebook content available"

            code_cells = []
            markdown_cells = []

            for cell in self.notebook_content.cells:
                if cell.cell_type == 'code':
                    code_cells.append(cell.source)
                elif cell.cell_type == 'markdown':
                    markdown_cells.append(cell.source)

            analysis = [
                "Notebook Analysis:",
                f"Total cells: {len(self.notebook_content.cells)}",
                f"Code cells: {len(code_cells)}",
                f"Markdown cells: {len(markdown_cells)}",
                "\nCode Overview:",
                *code_cells,
                "\nMarkdown Overview:",
                *markdown_cells
            ]

            return "\n".join(analysis)
        except Exception as e:
            raise Exception(f"Error analyzing notebook: {str(e)}")

    def _enhance_context(self, context: str) -> str:
        """Use OpenAI to enhance and structure the context."""
        try:
            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping to analyze and structure data science context information."},
                {"role": "user", "content": f"Please analyze and structure the following data science context information, highlighting key insights and potential challenges. Include sections for DataFrame Analysis, Code Analysis, and LLM Interpretation:\n\n{context}"}
            ])
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error enhancing context with OpenAI: {str(e)}")

def save_insights(context, file_path='llm_context.txt'):
    """Save all insights to a single file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(context)

if __name__ == "__main__":
    builder = DataContextBuilder()

    # Get user input
    train_path = input("Enter the path to your training data CSV file: ")
    has_test = input("Do you have separate test data? (yes/no): ").lower() == 'yes'
    test_path = input("Enter the path to your test data CSV file: ") if has_test else None
    notebook_path = input("Enter the path to your IPython notebook (.ipynb) file: ")

    # Load data and notebook
    builder.load_data(train_path, test_path)
    builder.load_notebook(notebook_path)

    # Generate context
    context = builder.generate_context()

    # Save all insights
    save_insights(context)

    print("Context has been generated and saved to 'llm_context.txt'")
    print("\nGenerated context preview:")
    print(context[:500] + "...")
