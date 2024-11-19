import nbformat
import os
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import json

class GeminiNotebookConverter:
    """Converts Jupyter notebooks to deployable code using Google's Gemini API"""
    
    def __init__(self, notebook_path: str):
        load_dotenv()
        self.notebook_path = notebook_path
        
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        
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
    
    async def generate_deployment_code(self, notebook_code: str) -> Dict[str, str]:
        """Use Gemini to analyze notebook and generate deployment code"""
        
        prompt = f"""You are an expert ML engineer. Analyze this Jupyter notebook code and generate production-ready deployment files.
        The notebook contains ML model code: 

        {notebook_code}

        Generate these files:
        1. model.py - Clean model definition and utilities
        2. api.py - FastAPI API with proper endpoints
        3. requirements.txt - All required packages
        4. README.md - Instructions for deployment
        
        For each file, start with ### Filename ### and then provide the content.
        Follow these guidelines:
        - Code should be well-organized and documented
        - Include proper error handling
        - Follow best practices for ML deployment
        - Have appropriate API endpoints
        - Include logging and monitoring
        """

        response = await self.model.generate_content_async(prompt)
        
        try:
            # Parse response and extract files
            content = response.text
            files = {}
            current_file = None
            current_content = []
            
            for line in content.split('\n'):
                if line.startswith('### ') and line.endswith(' ###'):
                    # Save previous file if exists
                    if current_file:
                        files[current_file] = '\n'.join(current_content)
                        current_content = []
                    # Start new file
                    current_file = line.strip('# ').lower()
                else:
                    current_content.append(line)
                    
            # Save last file
            if current_file:
                files[current_file] = '\n'.join(current_content)
                
            return files
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    async def generate_frontend(self) -> str:
        """Generate React frontend code using Gemini"""
        
        prompt = """Generate a React frontend component for an ML model API with:
        1. File upload for input data
        2. Display for model predictions
        3. Error handling
        4. Loading states
        5. Clean UI using shadcn/ui components
        
        Return only the React component code. Use Tailwind CSS for styling."""

        response = await self.model.generate_content_async(prompt)
        return response.text

    async def save_files(self, output_dir: str):
        """Generate and save all deployment files"""
        try:
            # Read notebook
            notebook_code = self.read_notebook()
            
            # Generate backend code
            files = await self.generate_deployment_code(notebook_code)
            if not files:
                raise Exception("Failed to generate deployment code")
            
            # Generate frontend code
            frontend_code = await self.generate_frontend()
            files['frontend.jsx'] = frontend_code
            
            # Save all files
            os.makedirs(output_dir, exist_ok=True)
            for filename, content in files.items():
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write(content)
                    
            print(f"Successfully generated deployment files in {output_dir}")
            
        except Exception as e:
            print(f"Error generating deployment files: {e}")

async def deploy_notebook(notebook_path: str, output_dir: str = 'deployment'):
    """Helper function to convert and deploy a notebook"""
    converter = GeminiNotebookConverter(notebook_path)
    await converter.save_files(output_dir)