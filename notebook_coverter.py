import nbformat
import os
from typing import Dict, List, Optional
import anthropic
from dotenv import load_dotenv
import json

class AINotebookConverter:
    """Converts Jupyter notebooks to deployable code using Claude"""
    
    def __init__(self, notebook_path: str):
        load_dotenv()
        self.notebook_path = notebook_path
        self.client = anthropic.Anthropic()
        
    def read_notebook(self) -> str:
        """Read and extract code from notebook"""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        # Extract only code cells
        code_cells = []
        for cell in nb.cells:
            if cell.cell_type == 'code':
                code_cells.append(cell.source)
                
        return '\n\n'.join(code_cells)
    
    def generate_deployment_code(self, notebook_code: str) -> Dict[str, str]:
        """Use Claude to analyze notebook and generate deployment code"""
        
        prompt = f"""As an AI expert in ML deployment, analyze this Jupyter notebook code and generate production-ready deployment files.
        The notebook contains ML model code: 

        {notebook_code}

        Generate these files:
        1. model.py - Clean model definition and utilities
        2. api.py - FastAPI API with proper endpoints
        3. requirements.txt - All required packages
        4. README.md - Instructions for deployment
        
        Return a JSON with keys 'model.py', 'api.py', 'requirements.txt', and 'README.md' containing the file contents.
        
        The code should:
        - Be well-organized and documented
        - Include proper error handling
        - Follow best practices for ML deployment
        - Have appropriate API endpoints
        - Include logging and monitoring
        """

        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0,
            system="You are an expert ML engineer specializing in deploying ML models to production. Generate only valid Python code and configuration files.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        try:
            # Extract JSON from response
            response_text = message.content[0].text
            # Find JSON block
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def generate_frontend(self) -> str:
        """Generate React frontend code using Claude"""
        
        prompt = """Generate a React frontend component for an ML model API with:
        1. File upload for input data
        2. Display for model predictions
        3. Error handling
        4. Loading states
        5. Clean UI using shadcn/ui components
        
        Return only the React component code."""

        message = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0,
            system="You are an expert frontend developer. Generate only valid React code using shadcn/ui components.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return message.content[0].text

    def save_files(self, output_dir: str):
        """Generate and save all deployment files"""
        try:
            # Read notebook
            notebook_code = self.read_notebook()
            
            # Generate backend code
            files = self.generate_deployment_code(notebook_code)
            if not files:
                raise Exception("Failed to generate deployment code")
            
            # Generate frontend code
            frontend_code = self.generate_frontend()
            files['frontend.jsx'] = frontend_code
            
            # Save all files
            os.makedirs(output_dir, exist_ok=True)
            for filename, content in files.items():
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write(content)
                    
            print(f"Successfully generated deployment files in {output_dir}")
            
        except Exception as e:
            print(f"Error generating deployment files: {e}")

def deploy_notebook(notebook_path: str, output_dir: str = 'deployment'):
    """Helper function to convert and deploy a notebook"""
    converter = AINotebookConverter(notebook_path)
    converter.save_files(output_dir)