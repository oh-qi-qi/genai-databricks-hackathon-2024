# Databricks notebook source
# MAGIC %pip install --upgrade pip
# MAGIC
# MAGIC %pip install -v mlflow==2.16.2 transformers==4.36.1 llama-index==0.11.17 pydantic==2.7.4 accelerate==0.25.0 PyPDF2==3.0.1 PyMuPDF==1.24.11 sentencepiece==0.1.99 pdfminer.six==20240706 sentence-transformers==2.2.2 databricks-vectorsearch==0.42 langchain==0.3.3 langchain-community==0.3.2 flashrank==0.2.9 rich==13.9.2 fuzzywuzzy==0.18.0 python-Levenshtein==0.26.0 inflection==0.5.1 cloudpickle==2.0.0 langchain-core==0.3.10 langsmith==0.1.132 databricks-sdk==0.34.0 inflection==0.5.1
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import time
import re
import io
from PyPDF2 import PdfReader
import warnings
# Handle potential import error with accelerate
try:
    pass
except ImportError as e:
    print(f"Warning: {e}. MLU support might not be available in this environment.")

#parse PDF bytes using PyPDF2
def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        parsed_content = [page_content.extract_text() for page_content in reader.pages]
        return "\n".join(parsed_content)
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None 

def pprint(obj):
  import pprint
  pprint.pprint(obj, compact=True, indent=1, width=100)

def index_exists(vsc, endpoint_name, index_full_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
          print(f'Unexpected error describing the index. This could be a permission issue.')
          raise e
  return False
