import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Databricks environment variables
DATABRICKS_URL = os.getenv('DATABRICKS_URL')
TOKEN = os.getenv('DATABRICKS_TOKEN')
DATABRICKS_WAREHOUSE_ID = os.getenv('DATABRICKS_WAREHOUSE_ID')

# Catalog, schema, and volume settings
catalog_name = "llm_workspace"
schema_name = "default"
volume_name = "regubim-ai-volume"
dataset_location = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

# Set the DBFS API endpoint for listing files
#volume_endpoint = f"{DATABRICKS_URL}/api/2.1/unity-catalog/volumes/{catalog_name}.{schema_name}.{volume_name}"

#endpoint = f"{DATABRICKS_URL}/api/2.1/unity-catalog/providers"
#
## Make the API request to list files in the dataset_location
#response = requests.get(
#    endpoint,
#    headers={"Authorization": f"Bearer {TOKEN}"},
#    params={"path": dataset_location}
#)
#
## Check the response
#if response.status_code == 200:
#    print(response.json())
##    files = response.json().get("files", [])
##    print(f"Files in {dataset_location}:")
##    for file in files:
##        print(f"File Name: {file['path']}, Size: {file['file_size']} bytes")
#else:
#    print(f"Failed to list files. Status code: {response.status_code}, Error: {response.text}")

