import click
import asyncio
import os
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.command()
@click.argument('notebook_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='deployment', help='Output directory for deployment files')
def main(notebook_path: str, output: str):
    """Convert a Jupyter notebook into a deployable API with frontend using Gemini"""
    
    try:
        console.print(Panel.fit(
            "🚀 Starting notebook conversion...",
            title="ML Model Deployment Generator (Gemini-powered)",
            border_style="blue"
        ))
        
        # Validate input
        if not notebook_path.endswith('.ipynb'):
            raise click.BadParameter('Input file must be a Jupyter notebook (.ipynb)')
        
        # Convert notebook using asyncio
        from main import deploy_notebook
        asyncio.run(deploy_notebook(notebook_path, output))
        
        # Success message
        console.print(Panel.fit(
            f"""✅ Successfully generated deployment files in '{output}' directory!
            
            To deploy:
            1. cd {output}
            2. pip install -r requirements.txt
            3. uvicorn api:app --reload
            
            The API will be available at http://localhost:8000
            """,
            title="Deployment Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel.fit(
            f"❌ Error: {str(e)}",
            title="Conversion Failed",
            border_style="red"
        ))

if __name__ == '__main__':
    main()