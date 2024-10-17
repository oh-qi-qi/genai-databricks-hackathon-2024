# Databricks notebook source
import os

catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
schema_name = spark.catalog.currentDatabase()
working_directory = os.getcwd()
volume_name = "regubim-ai-volume"
dataset_location = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"

print(f"Catalog Name: {catalog_name}")
print(f"Schema Name: {schema_name}")
print(f"Working Directory: {working_directory}")
print(f"Volume Name: {volume_name}")
print(f"Dataset Location: {dataset_location}")

files = dbutils.fs.ls(dataset_location)

# Print the files and folders in the volume
for file in files:
    print(file.name)

# COMMAND ----------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import time
import re
import io
from PyPDF2 import PdfReader
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import json

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

# display result nicely in panel
def print_nested_dict_display(data):
    console = Console()

    def format_value(value):
        if isinstance(value, str):
            try:
                # Try to parse as JSON first
                json_data = json.loads(value)
                return Markdown(f"```json\n{json.dumps(json_data, indent=2)}\n```")
            except json.JSONDecodeError:
                # If not JSON, treat as Markdown
                return Markdown(value)
        elif isinstance(value, dict):
            return Markdown(f"```json\n{json.dumps(value, indent=2)}\n```")
        else:
            return str(value)

    for key, value in data.items():
        formatted_value = format_value(value)
        panel = Panel(formatted_value, title=key, expand=False)
        console.print(panel)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

def index_exists(vsc, endpoint_name, index_full_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
          print(f'Unexpected error describing the index. This could be a permission issue.')
          raise e
  return False

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

def create_vs_endpoint(vs_endpoint_name):
    vsc = VectorSearchClient()

    # check if the endpoint exists
    if vs_endpoint_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
        vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

    # check the status of the endpoint
    wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
    print(f"Endpoint named {vs_endpoint_name} is ready.")

def create_vs_index(vs_endpoint_name, vs_index_fullname, source_table_fullname, source_col):
    #create compute endpoint
    vsc = VectorSearchClient()
    create_vs_endpoint(vs_endpoint_name)
    
    # create or sync the index
    if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
        print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
        
        vsc.create_delta_sync_index(
            endpoint_name=vs_endpoint_name,
            index_name=vs_index_fullname,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED", #Sync needs to be manually triggered
            primary_key="id",
            embedding_source_column=source_col,
            embedding_model_endpoint_name="databricks-bge-large-en"
        )
        
        # vsc.create_delta_sync_index(
        #     endpoint_name=vs_endpoint_name,
        #     index_name=vs_index_fullname,
        #     source_table_name=source_table_fullname,
        #     pipeline_type="TRIGGERED", #Sync needs to be manually triggered
        #     primary_key="id",
        #     embedding_dimension=1024, #Match your model embedding size (bge)
        #     embedding_vector_column="embedding"
        # )

    else:
        #Trigger a sync to update our vs content with the new data saved in the table
        vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

    #Let's wait for the index to be ready and all our embeddings to be created and indexed
    wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)
