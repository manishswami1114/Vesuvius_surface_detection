import nbformat
from nbconvert import PythonExporter
from pathlib import Path
import sys

def nb_to_py(nb_path, out_path=None):
    nb_path = Path(nb_path)
    if out_path is None:
        out_path = nb_path.with_suffix('.py')
    nb = nbformat.read(str(nb_path), as_version=4)
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    with open(out_path, 'w') as f:
        f.write("# Converted from notebook: {}\n".format(nb_path.name))
        f.write(source)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/nb_to_script.py path/to/notebook.ipynb [out.py]")
        sys.exit(1)
    nb = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print("Writing:", nb_to_py(nb, out))
