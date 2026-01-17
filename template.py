import os
from pathlib import Path

list_of_files_and_directories = [
    "src/constants/__init__.py",
    "src/components/__init__.py",
    "src/components/data_preprocessing.py",
    "src/components/data_vectorization.py",
    "src/pipelines/training_pipeline.py",
    "src/logger/__init__.py",
    "tests/__init__.py",
    "tests/test_preprocessing.py",
    "tests/test_vectorization.py",
    "tests/test_pipeline.py",
    "./notebooks/00_datasets_preprocessing.ipynb",
    "./notebooks/01_content_based_filtering.ipynb",
    "server.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
]

for filepath in list_of_files_and_directories:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"{filepath} already exists")