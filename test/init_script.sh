#!/bin/bash

# Upgrade pip and install required packages
/databricks/python/bin/pip install --upgrade pip

/databricks/python/bin/pip install -v mlflow==2.16.2 transformers==4.36.1 llama-index==0.11.17 \
pydantic==2.7.4 accelerate==0.25.0 PyPDF2==3.0.1 PyMuPDF==1.24.11 sentencepiece==0.1.99 \
pdfminer.six==20240706 sentence-transformers==2.2.2 databricks-vectorsearch==0.42 langchain==0.3.3 \
langchain-community==0.3.2 flashrank==0.2.9 rich==13.9.2 fuzzywuzzy==0.18.0 python-Levenshtein==0.26.0 \
inflection==0.5.1 cloudpickle==2.0.0 langchain-core==0.3.10 langsmith==0.1.132 databricks-sdk==0.34.0
