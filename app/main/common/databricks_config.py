import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Databricks environment variables
DATABRICKS_HOST = os.getenv('DATABRICKS_HOST')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
DATABRICKS_WAREHOUSE_ID = os.getenv('DATABRICKS_WAREHOUSE_ID')
DATABRICKS_LLM_SERVING_ENDPOINT = os.getenv('DATABRICKS_LLM_SERVING_ENDPOINT')

# Catalog, schema, and volume settings
catalog_name = "llm_workspace"
schema_name = "default"
volume_name = "regubim-ai-volume"
dataset_location = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

