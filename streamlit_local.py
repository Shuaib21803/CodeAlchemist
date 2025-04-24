import pandas as pd
import nbformat
import os
import json
import uuid
import joblib
import tempfile
import base64
import shutil
import streamlit as st
from typing import Dict, Any, List, Optional, Union, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

class DataContextBuilder:
    def __init__(self):
        """Initialize the context builder."""
        self.train_df = None
        self.test_df = None
        self.notebook_content = None
        self.user_responses = {}
        self.model = None
        self.model_columns = None
        self.model_type = None
        self.task_type = None  # 'classification' or 'regression'
        self.target_column = None

    def load_data(self, train_file, test_file=None) -> None:
        """Load train and optionally test data from uploaded files."""
        try:
            self.train_df = pd.read_csv(train_file)
            if test_file:
                self.test_df = pd.read_csv(test_file)
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def load_notebook(self, notebook_file) -> None:
        """Load and parse IPython notebook from uploaded file."""
        try:
            notebook_content = notebook_file.read()
            self.notebook_content = nbformat.reads(notebook_content.decode('utf-8'), as_version=4)
        except Exception as e:
            raise Exception(f"Error loading notebook: {str(e)}")
    
    def load_model(self, model_file) -> None:
        """Load the pickled sklearn model from uploaded file."""
        try:
            # Save to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(model_file.read())
                tmp_path = tmp.name
            
            # Load the model from the temporary file
            self.model = joblib.load(tmp_path)
            
            # Remove the temporary file
            os.unlink(tmp_path)
            
            # Determine model type and extract feature names
            self._analyze_model()
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _analyze_model(self) -> None:
        """Analyze the loaded model to determine its type and extract column names."""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Loaded object is not a valid model (missing predict method)")
        
        # Determine if it's a classification or regression model
        if hasattr(self.model, 'predict_proba'):
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
            
        # Try to extract feature names
        if hasattr(self.model, 'feature_names_in_'):
            self.model_columns = list(self.model.feature_names_in_)
        else:
            # For pipeline or more complex models
            self._extract_columns_from_pipeline()
            
        # Get the specific model type
        self.model_type = type(self.model).__name__
        
        # Try to determine target column from notebook analysis if available
        if self.notebook_content:
            self._extract_target_from_notebook()

    def _extract_columns_from_pipeline(self) -> None:
        """Extract column names from a pipeline or complex model."""
        # This is a simplified approach - in a real system you'd need more robust extraction
        try:
            if hasattr(self.model, 'steps'):
                # It's a pipeline
                for name, step in self.model.steps:
                    if hasattr(step, 'feature_names_in_'):
                        self.model_columns = list(step.feature_names_in_)
                        return
            
            # If we couldn't extract columns, use the training data columns
            if self.train_df is not None:
                # Assume all columns except the last one are features
                self.model_columns = list(self.train_df.columns[:-1])
        except:
            # Fallback: if we can't determine columns, we'll ask the user later
            self.model_columns = None

    def _extract_target_from_notebook(self) -> None:
        """Try to extract target column name from notebook code."""
        target_patterns = ["y =", "target =", ".target", "y_train ="]
        
        for cell in self.notebook_content.cells:
            if cell.cell_type == 'code':
                code = cell.source
                lines = code.split('\n')
                for line in lines:
                    for pattern in target_patterns:
                        if pattern in line:
                            # Extract variable name after the '=' sign
                            parts = line.split('=')
                            if len(parts) > 1 and pattern == "y =" or pattern == "target =" or pattern == "y_train =":
                                # This is a very basic extraction and would need to be more robust
                                potential_target = parts[1].strip()
                                if "df[" in potential_target:
                                    # Extract column name from df['column'] or df["column"] pattern
                                    col_start = potential_target.find("[") + 1
                                    col_end = potential_target.find("]")
                                    if col_start > 0 and col_end > col_start:
                                        col_name = potential_target[col_start:col_end].strip("'\"")
                                        self.target_column = col_name
                                        return

    def generate_context(self) -> str:
        """Generate context for LLM from all available information."""
        try:
            if self.train_df is None:
                raise ValueError("Training data not loaded")

            context_parts = []

            # Add model information
            if self.model is not None:
                context_parts.append("=== MODEL ANALYSIS ===")
                context_parts.append(f"Model Type: {self.model_type}")
                context_parts.append(f"Task Type: {self.task_type}")
                if self.model_columns:
                    context_parts.append(f"Model Features: {', '.join(self.model_columns)}")
                if self.target_column:
                    context_parts.append(f"Target Column: {self.target_column}")

            # Add data analysis information
            context_parts.append("\n=== TRAINING DATA ANALYSIS ===")
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
                return full_context + "\n\n" + enhanced_context
            except Exception as e:
                st.warning(f"Could not enhance context with OpenAI: {str(e)}")
                return full_context

        except Exception as e:
            raise Exception(f"Error generating context: {str(e)}")

    def _get_dataframe_info(self, df: pd.DataFrame, dataset_type: str) -> str:
        """Get comprehensive information about a dataframe."""
        try:
            buffer = []
            buffer.append(f"{dataset_type} Dataset Shape: {df.shape}")
            
            # DataFrame info
            buffer.append(f"\nColumn Names: {list(df.columns)}")
            buffer.append(f"\nData Types:\n{df.dtypes.to_string()}")
            
            # Missing values
            buffer.append(f"\nMissing Values:\n{df.isnull().sum().to_string()}")
            
            # Basic statistics
            buffer.append(f"\nDescriptive Statistics (Numeric Columns):")
            buffer.append(df.describe().to_string())
            
            # Categorical columns analysis
            cat_columns = df.select_dtypes(include=['object', 'category']).columns
            if not cat_columns.empty:
                buffer.append("\nCategorical Columns Value Counts:")
                for col in cat_columns:
                    buffer.append(f"\n{col}:\n{df[col].value_counts().head(10).to_string()}")
            
            return "\n".join(buffer)
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
                "\nCode Overview (truncated to key parts):"
            ]
            
            # Add only key code parts (to avoid making the context too large)
            key_keywords = ['import', 'model', 'train', 'fit', 'predict', 'score', 'accuracy', 'classification', 'regression']
            filtered_code = []
            
            for code in code_cells:
                if any(keyword in code for keyword in key_keywords):
                    filtered_code.append(code)
            
            analysis.extend(filtered_code[:10])  # Limit to first 10 relevant code cells
            
            return "\n".join(analysis)
        except Exception as e:
            return f"Error analyzing notebook: {str(e)}"

    def _enhance_context(self, context: str) -> str:
        """Use OpenAI to enhance and structure the context."""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping to analyze and structure data science context information."},
                    {"role": "user", "content": f"Please analyze and structure the following data science context information, highlighting key insights and potential challenges. Focus on aspects needed to create a Flask API for serving this model. Include sections for Model Analysis, DataFrame Analysis, and API Requirements:\n\n{context}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error enhancing context with OpenAI: {str(e)}")

    def generate_api_code(self) -> dict:
        """Generate Flask API code for serving the model."""
        if not self.model:
            raise ValueError("Model not loaded")
            
        context = self.generate_context()
        
        # Generate Flask API code using OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",  # Using a more capable model for code generation
                messages=[
                    {"role": "system", "content": """You are an expert Flask API developer specialized in serving machine learning models. 
                Create complete, production-ready Flask code based on the context information.
                Your code must include proper error handling, input validation, and documentation.
                Ensure the API accepts both JSON input and form submissions with proper handling for each case.
                Include HTML templates for a simple user interface.
                Use specific code structure for the predict route that handles both JSON and form data."""},
                {"role": "user", "content": f"""Using this context information, create a complete Flask API to serve the described model:
                
                {context}
                
                Provide the following in your response as a JSON object with these keys:
                1. 'app_code': The main Flask application code (Absolute imports, don't use relative imports)
                2. 'templates': A dictionary mapping template names to their HTML content
                3. 'static_files': A dictionary mapping static file names to their content
                4. 'requirements': A list of required Python packages
                5. 'api_documentation': Markdown documentation describing how to use the API
                
                IMPORTANT REQUIREMENTS:
                
                1. Load model files, like this:
                   - Use `model = joblib.load('ml_model_api/model.pkl')` ## strict naming here
                   - Use `model_columns = joblib.load('ml_model_api/model.pkl')` ## strict naming here
                
                2. Structure the predict route to handle BOTH JSON API requests AND form submissions like this:
                @app.route('/predict', methods=['POST'])
                def predict() -> Any:
  
                    Handle prediction requests from JSON data or form submission.
                    :return: JSON or rendered template with predicted sales
                    
                    # Handle JSON requests (API)
                    if request.is_json:
                        data = request.get_json()
                        
                        # Validate and parse inputs
                        try:
                            # Input validation code here
                        except (TypeError, ValueError):
                            return jsonify, 400
                        
                        # Perform prediction with validated inputs
                        try:
                            # Prediction code here
                        except Exception as e:
                            return jsonify
                        
                        return jsonify
                    
                    # Handle form submissions (Web UI)
                    else:
                        data = request.form
                        
                        # Validate and parse inputs
                        try:
                            # Input validation code here
                        except (TypeError, ValueError):
                            return render_template('index.html', error='Invalid input values')
                        
                        # Perform prediction with validated inputs
                        try:
                            # Prediction code here
                        except Exception as e:
                            
                        
                        # Return the template with the prediction result
                        return render_template('index.html', prediction_result=prediction)

                
                3. Create a user-friendly web interface in the index.html template that:
                   - Has a clear form with labeled inputs for all required features
                   - Shows prediction results and any errors on the same page
                   - Uses basic CSS for formatting (keep it simple)
                
                4. Include proper handling for both API usage (with curl/Postman) and browser form usage
                """}
            ]
        )
            
            # Parse the response as JSON
            content = response.choices[0].message.content
            
            # Extract the JSON part from the response (in case there's text before/after)
            try:
                return json.loads(content)
            except:
                # If direct parsing fails, try to find JSON within the text
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]
                    return json.loads(json_content)
                else:
                    raise ValueError("Could not extract valid JSON from the API generation response")
                
        except Exception as e:
            raise Exception(f"Error generating API code: {str(e)}")

def save_api_files(api_code, save_dir):
    """Save the generated API code and templates to disk."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'static'), exist_ok=True)
    
    # Save main Flask app code
    with open(os.path.join(save_dir, 'app.py'), 'w') as f:
        f.write(api_code['app_code'])
    
    # Save templates
    for template_name, content in api_code['templates'].items():
        with open(os.path.join(save_dir, 'templates', template_name), 'w') as f:
            f.write(content)
    
    # Save static files
    for file_name, content in api_code['static_files'].items():
        with open(os.path.join(save_dir, 'static', file_name), 'w') as f:
            f.write(content)
    
    # Save requirements
    with open(os.path.join(save_dir, 'requirements.txt'), 'w') as f:
        f.write('\n'.join(api_code['requirements']))
    
    # Save API documentation
    with open(os.path.join(save_dir, 'API_DOCUMENTATION.md'), 'w') as f:
        f.write(api_code['api_documentation'])
    
    # Create a README.md file
    with open(os.path.join(save_dir, 'README.md'), 'w') as f:
        f.write(f"""# Generated ML Model API

This API was automatically generated to serve your machine learning model.

## Getting Started

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the API:
   ```
   python app.py
   ```

3. Access the API at http://localhost:5000

## API Documentation

{api_code['api_documentation']}
""")

def create_download_link(file_path, file_name):
    """Create a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

def create_zip_download(folder_path, zip_name):
    """Create a downloadable zip file from a folder."""
    temp_zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    shutil.make_archive(temp_zip_path.replace('.zip', ''), 'zip', folder_path)
    
    with open(temp_zip_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{zip_name}">Download Complete API as ZIP</a>'
    return href

def main():
    st.set_page_config(page_title="ML Model Hosting", page_icon="🤖", layout="wide")
    
    st.title("ML Model Hosting Service")
    st.write("Upload your model, data, and notebook to generate a Flask API")
    
    # Session state to track progress
    if 'model_processed' not in st.session_state:
        st.session_state.model_processed = False
    if 'api_generated' not in st.session_state:
        st.session_state.api_generated = False
    if 'api_code' not in st.session_state:
        st.session_state.api_code = None
    if 'api_dir' not in st.session_state:
        st.session_state.api_dir = None
    if 'context_builder' not in st.session_state:
        st.session_state.context_builder = DataContextBuilder()
    
    # File upload section
    with st.expander("Upload Files", expanded=not st.session_state.model_processed):
        col1, col2 = st.columns(2)
        
        with col1:
            train_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
            test_file = st.file_uploader("Upload Test Data (CSV, optional)", type=['csv'])
        
        with col2:
            notebook_file = st.file_uploader("Upload Jupyter Notebook (.ipynb)", type=['ipynb'])
            model_file = st.file_uploader("Upload Trained Model (.pkl)", type=['pkl'])
        
        if st.button("Process Files", disabled=not (train_file and notebook_file and model_file)):
            with st.spinner("Processing files..."):
                try:
                    # Create a new context builder
                    builder = DataContextBuilder()
                    
                    # Load all files
                    builder.load_data(train_file, test_file)
                    builder.load_notebook(notebook_file)
                    builder.load_model(model_file)
                    
                    # Store in session state
                    st.session_state.context_builder = builder
                    st.session_state.model_processed = True
                    
                    st.success("Files processed successfully!")
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
    
    # Model analysis section
    if st.session_state.model_processed:
        with st.expander("Model Analysis", expanded=True):
            builder = st.session_state.context_builder
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Information")
                st.write(f"**Model Type:** {builder.model_type}")
                st.write(f"**Task Type:** {builder.task_type}")
                
                if builder.target_column:
                    st.write(f"**Target Column:** {builder.target_column}")
                
                st.subheader("Dataset Information")
                st.write(f"**Training Data Shape:** {builder.train_df.shape}")
                if builder.test_df is not None:
                    st.write(f"**Test Data Shape:** {builder.test_df.shape}")
            
            with col2:
                st.subheader("Features")
                if builder.model_columns:
                    st.write(f"**Number of Features:** {len(builder.model_columns)}")
                    if len(builder.model_columns) <= 10:
                        st.write(f"**Features:** {', '.join(builder.model_columns)}")
                    else:
                        st.write(f"**First 10 Features:** {', '.join(builder.model_columns[:10])}...")
                
                st.subheader("Data Preview")
                st.dataframe(builder.train_df.head())
            
            # Generate API button
            if st.button("Generate API"):
                with st.spinner("Generating API code..."):
                    try:
                        api_code = builder.generate_api_code()
                        
                        # Create a unique directory for this API
                        api_dir = os.path.join(tempfile.gettempdir(), f"ml_api_{uuid.uuid4().hex}")
                        os.makedirs(api_dir, exist_ok=True)
                        
                        # Save model and columns
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            joblib.dump(builder.model, os.path.join(api_dir, 'model.pkl'))
                        
                        if builder.model_columns:
                            joblib.dump(builder.model_columns, os.path.join(api_dir, 'model_columns.pkl'))
                        
                        # Save API files
                        save_api_files(api_code, api_dir)
                        
                        # Store in session state
                        st.session_state.api_generated = True
                        st.session_state.api_code = api_code
                        st.session_state.api_dir = api_dir
                        
                        st.success("API generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating API: {str(e)}")
    
    # API details section
    if st.session_state.api_generated and st.session_state.api_code:
        with st.expander("Generated API", expanded=True):
            api_code = st.session_state.api_code
            api_dir = st.session_state.api_dir
            
            st.subheader("API Files")
            
            # Tabs for different parts of the generated API
            tab1, tab2, tab3, tab4 = st.tabs(["Flask App", "Templates", "Documentation", "Download"])
            
            with tab1:
                st.code(api_code['app_code'], language='python')
            
            with tab2:
                template_select = st.selectbox("Select template", 
                                              list(api_code['templates'].keys()))
                if template_select:
                    st.code(api_code['templates'][template_select], language='html')
            
            with tab3:
                st.markdown(api_code['api_documentation'])
            
            with tab4:
                st.markdown("### Download Files")
                
                # Create zip download link
                zip_link = create_zip_download(api_dir, 'ml_model_api.zip')
                st.markdown(zip_link, unsafe_allow_html=True)
                
                st.markdown("### How to Run")
                st.code("""
# Install requirements
pip install -r requirements.txt

# Run the Flask app
python app.py

# Your API will be available at http://localhost:5000
                """, language='bash')
                
                st.markdown("### Production Deployment")
                st.info("""
                To deploy this API to production:
                1. Set up a web server (Nginx/Apache) with your domain
                2. Use Gunicorn or uWSGI to serve the Flask app
                3. Set up SSL certificates for HTTPS
                4. Consider containerizing with Docker for easier deployment
                """)

if __name__ == "__main__":
    main()