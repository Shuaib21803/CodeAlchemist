import nbformat
from nbconvert import PythonExporter

def convert(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    exporter = PythonExporter(template_name="python")
    script, _ = exporter.from_notebook_node(notebook_content)

    output_path = notebook_path.replace(".ipynb", ".py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"Converted {notebook_path} to {output_path}")