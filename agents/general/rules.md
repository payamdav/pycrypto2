# General Rules

This document defines mandatory rules that all AI agents **must** follow when performing any task in this repository. Rules will be added progressively over time.

---

## Dependency Management

### Jupyter Notebooks

When creating or editing a Jupyter notebook (`.ipynb`), **all** packages required by the code—whether imported directly or used as transitive dependencies—must be installed inline using `%pip install` at the top of the notebook before any import statements.

Example pattern (first code cell):

```python
%pip install pandas numpy scikit-learn
```

> Do **not** assume packages are pre-installed. Always include an explicit `%pip install` cell so the notebook is self-contained and reproducible.

### Python Scripts and Modules

When creating any Python file (`.py`) that depends on external packages, a `requirements.txt` file **must** be placed in the same directory as the script (or at the root of the relevant sub-folder if multiple scripts share the same dependencies).

The `requirements.txt` must list every external package required to run the script(s) in that folder.

---

## Repository Access in Jupyter Notebooks

When a Jupyter notebook imports or uses any packages, modules, or other parts of **this project**, the repository **must** be cloned inside the Jupyter server environment so that the project code is available at runtime. Without this step the notebook will fail to resolve local imports.

If the notebook is created on the **main** branch, use the following pattern at the top of the notebook (before any project imports):

```python
import os
import sys

# 1. Define your GitHub repository details
REPO_URL = "https://github.com/payamdav/pycrypto.git"
REPO_NAME = "pycrypto"

# 2. Clone the repo if it hasn't been cloned yet
if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}

# 3. Add the cloned repository root to the Python path
REPO_PATH = os.path.abspath(REPO_NAME)
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
```

---

## Package Documentation

Every package that resides in `packages/` **must** have a corresponding documentation file in `agents/packages/`. The documentation file should be named after the package (e.g., `agents/packages/<package_name>.md`) and must describe the package's purpose, public API, usage examples, and any important conventions or constraints.

It is the **responsibility of the package developer** to create and maintain this documentation file whenever a new package is added or an existing package is significantly changed. Agents and other developers rely on these files as the authoritative reference for each package.

---

## Writing Style

All text written by agents — documents, descriptions, docstrings, comments, READMEs, or any other written output — must be **as short as possible** while remaining meaningful, fully covering the topic, easy to read, and well-structured. No padding, no redundancy, no filler sentences. If something can be said in one sentence instead of three, use one.

---

*End of rules — additional rules will be appended below as needed.*
