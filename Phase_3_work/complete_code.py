"""Convert a Python script with # %% markers to a Jupyter notebook (.ipynb)."""

import re
import json
import sys
from pathlib import Path


def py_to_notebook(py_path: str, ipynb_path: str = None):
    """Convert a .py file with # %% cell markers to .ipynb format."""

    py_path = Path(py_path)
    if ipynb_path is None:
        ipynb_path = py_path.with_suffix('.ipynb')
    else:
        ipynb_path = Path(ipynb_path)

    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    cells = []
    current_cell_type = 'code'
    current_lines = []

    for line in lines:
        stripped = line.strip()

        # Check for cell markers
        if stripped == '# %%' or stripped.startswith('# %% ') and '[markdown]' not in stripped:
            # Save previous cell
            if current_lines:
                cells.append(make_cell(current_cell_type, current_lines))
            current_cell_type = 'code'
            current_lines = []
            continue

        if '# %% [markdown]' in stripped:
            # Save previous cell
            if current_lines:
                cells.append(make_cell(current_cell_type, current_lines))
            current_cell_type = 'markdown'
            current_lines = []
            continue

        current_lines.append(line)

    # Save final cell
    if current_lines:
        cells.append(make_cell(current_cell_type, current_lines))

    # Build notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Converted: {py_path} -> {ipynb_path}")
    print(f"Total cells: {len(cells)}")
    code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
    md_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
    print(f"  Code cells: {code_cells}")
    print(f"  Markdown cells: {md_cells}")


def make_cell(cell_type: str, lines: list) -> dict:
    """Create a notebook cell from lines."""

    if cell_type == 'markdown':
        # Strip leading '# ' from each line for markdown
        processed = []
        for line in lines:
            if line.startswith('# '):
                processed.append(line[2:])
            elif line.strip() == '#':
                processed.append('')
            elif line.startswith('#') and not line.startswith('#!'):
                processed.append(line[1:].lstrip(' ') if len(line) > 1 else '')
            else:
                processed.append(line)

        # Remove leading/trailing blank lines
        while processed and not processed[0].strip():
            processed.pop(0)
        while processed and not processed[-1].strip():
            processed.pop()

        source = [line + '\n' for line in processed]
        if source:
            source[-1] = source[-1].rstrip('\n')

        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        }
    else:
        # Code cell - remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        source = [line + '\n' for line in lines]
        if source:
            source[-1] = source[-1].rstrip('\n')

        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        }


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    py_file = script_dir / 'phase3_instashap_analysis.py'
    ipynb_file = script_dir / 'Phase3_InstaSHAP_Analysis.ipynb'

    if not py_file.exists():
        print(f"ERROR: {py_file} not found!")
        sys.exit(1)

    py_to_notebook(str(py_file), str(ipynb_file))
    print(f"\nNotebook ready at: {ipynb_file}")
