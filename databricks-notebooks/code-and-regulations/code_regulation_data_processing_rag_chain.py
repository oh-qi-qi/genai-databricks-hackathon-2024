# Databricks notebook source
# MAGIC %md
# MAGIC #### Dependency and Installation

# COMMAND ----------

# MAGIC %run ./dependency_installation_setup

# COMMAND ----------

import os

catalog_name = spark.sql("SELECT current_catalog()").collect()[0][0]
schema_name = spark.catalog.currentDatabase()
working_directory = os.getcwd()
dataset_location = f"/Volumes/{catalog_name}/{schema_name}/regubim-ai-volume/"

print(f"Catalog Name: {catalog_name}")
print(f"Schema Name: {schema_name}")
print(f"Working Directory: {working_directory}")
print(f"Dataset Location: {dataset_location}")

files = dbutils.fs.ls(dataset_location)

# Print the files and folders in the volume
for file in files:
    print(file.name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Process Code and Regulation Documents

# COMMAND ----------

# Read the code and regulation documents and the write it as a delta table
table_name = f"{catalog_name}.{schema_name}.code_regulations_engineering_pdf_raw_text"

# read pdf files
df = (
        spark.read.format("binaryfile")
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.pdf")  # This ensures only PDF files are read
        .load(dataset_location)
        )

# save list of the files to table
df.write.mode("overwrite").saveAsTable(table_name)

display(df)

# COMMAND ----------

# Perform hierarchy chunking based on section
import io
import re
from typing import List, Dict
import ast

import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pyspark.sql.functions import col, explode, pandas_udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

# Define the schema for our chunks (sections)
chunk_schema = ArrayType(StructType([
    StructField("text", StringType(), True),            # Chunk text
    StructField("section_header", StringType(), True),  # Hierarchical section path
    StructField("document_name", StringType(), True),   # Document name
    StructField("chunk_id", IntegerType(), True),       # Chunk count identifier
    StructField("token_count", IntegerType(), True)     # Token count
]))

# Define headers globally (custom headers that might appear in documents)
headers = []

def parse_pdf(pdf_bytes: bytes) -> List[Dict[str, str]]:
    results = []
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        for page_layout in extract_pages(pdf_file):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    line_text = element.get_text().strip()
                    if line_text:
                        results.append({'text': line_text})
        return results
    except Exception as e:
        print(f"Error parsing PDF: {str(e)}")
        return []

class HierarchicalNodeParser:
    def __init__(self):
        pass

    def compile_header_patterns(self, custom_headers: List[str]):
        """
        Compile regex patterns for different section header formats, including headers
        that start after a newline with optional spaces.
        """
        header_patterns = [
            r'^\s*([1-9]\d*)\.\s+([A-Za-z].*)$',  # Single-level header (1. Introduction)
            r'^\s*([1-9]\d*\.\d+)\s+([A-Za-z].*)$',  # Two-level header (1.1 xxxx)
            r'^\s*([1-9]\d*\.\d+\.\d+)\s+([A-Za-z].*)$',  # Three-level header (1.1.1 xxxxx)
            r'\n\s*([1-9]\d*)\.\s+([A-Za-z].*)$',  # Header starting after newline with spaces (/n 1. Introduction)
            r'^(?:' + '|'.join(re.escape(header) for header in custom_headers) + r')$',  # Custom headers
        ]
        return [re.compile(pattern, re.MULTILINE | re.DOTALL) for pattern in header_patterns]

    def is_header(self, line: str, compiled_patterns) -> bool:
        """
        Check if a line matches any of the compiled header patterns.
        """
        for pattern in compiled_patterns:
            if pattern.match(line):
                return True
        return False

    def get_nodes_from_elements(self, elements: List[Dict[str, str]], document_name: str, custom_headers: List[str]) -> List[Dict[str, str]]:
        all_nodes = []
        # Split the elements into sections based on headers
        sections = self.split_elements_by_section(elements, custom_headers)
        chunk_id = 0  # Initialize chunk count

        # Set minimum and maximum token counts for chunks
        min_tokens = 50
        max_tokens = 500

        # Process each section
        for section_path, section_data in sections.items():
            # Combine text lines into a single string for each section
            section_text = '\n'.join(section_data['text']).strip()
            tokens = section_text.split()
            num_tokens = len(tokens)
            idx = 0  # Token index
            temp_chunks = []  # Temporary list to hold chunks before merging

            while idx < num_tokens:
                end_idx = min(idx + max_tokens, num_tokens)
                chunk_tokens = tokens[idx:end_idx]

                # Find the last period in the current chunk
                period_idx = None
                for i in range(len(chunk_tokens) - 1, -1, -1):
                    if any(chunk_tokens[i].endswith(punct) for punct in ('.', '!', '?')):
                        period_idx = i
                        break

                if period_idx is not None and idx + period_idx + 1 - idx >= min_tokens:
                    # If a period is found and chunk meets minimum token count
                    end_idx = idx + period_idx + 1
                else:
                    # Try to find the next period to meet minimum token count
                    for i in range(end_idx, num_tokens):
                        if any(tokens[i].endswith(punct) for punct in ('.', '!', '?')):
                            end_idx = i + 1
                            if end_idx - idx >= min_tokens:
                                break
                    else:
                        # If no period is found, and we are at the end, set end_idx to num_tokens
                        end_idx = num_tokens

                # Extract the chunk tokens and create the chunk text
                chunk_tokens = tokens[idx:end_idx]
                chunk_text = ' '.join(chunk_tokens)
                chunk_token_count = len(chunk_tokens)

                # Convert the string to a tuple using ast.literal_eval
                section_path_tuple = ast.literal_eval(f'[{section_path}]')
                # Extract the second element (the string part) from each tuple
                result_list = [section[1] for section in section_path_tuple]
                # Join the extracted headers into a single string
                result = ', '.join(result_list)

                
                # Append section header to the chunk text
                chunk_text = f"{result}\n{chunk_text}"

                # Append to temporary list
                temp_chunks.append({
                    'text': chunk_text,
                    'section_header': result,
                    'document_name': document_name,
                    'token_count': chunk_token_count
                })

                # Move the index to the end of the current chunk
                idx = end_idx

            # Merge short chunks with previous ones if necessary (only if in the same section)
            merged_chunks = []
            for chunk in temp_chunks:
                if merged_chunks and chunk['token_count'] < min_tokens and merged_chunks[-1]['section_header'] == chunk['section_header']:
                    # Merge with the previous chunk in the same section
                    prev_chunk = merged_chunks[-1]
                    prev_chunk['text'] += ' ' + chunk['text']
                    prev_chunk['token_count'] += chunk['token_count']
                else:
                    # Start a new chunk
                    merged_chunks.append(chunk)

            # Assign chunk IDs and collect chunks
            for chunk in merged_chunks:
                chunk_id += 1
                chunk['chunk_id'] = chunk_id
                all_nodes.append(chunk)

        return all_nodes

    def split_elements_by_section(self, elements: List[Dict[str, str]], custom_headers: List[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Split text elements into sections based on detected headers.
        Track section hierarchy
        """
        if custom_headers is None:
            custom_headers = []

        compiled_patterns = self.compile_header_patterns(custom_headers)

        sections = {}
        current_text = []
        section_stack = []
        default_section = "Unknown Section"

        for element in elements:
            line_text = element['text']
            stripped_line = line_text.strip()

            # Check if the line matches any header pattern
            if self.is_header(stripped_line, compiled_patterns):
                # Save the current accumulated text to the previous section
                if section_stack:
                    current_section_path = self.create_section_path(section_stack)
                    if current_section_path not in sections:
                        sections[current_section_path] = {'text': []}
                    sections[current_section_path]['text'].extend(current_text)
                    current_text = []

                # Register new section, even if it has no content yet
                section_title = stripped_line
                # Determine the level based on the section numbering
                level = section_title.count('.') if '.' in section_title else 1

                # Adjust the section stack to the current level
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, section_title))

                # Ensure the section is registered, even if it has no content
                current_section_path = self.create_section_path(section_stack)
                if current_section_path not in sections:
                    sections[current_section_path] = {'text': []}

            else:
                # It's not a header, accumulate text
                current_text.append(line_text)

        # After processing all elements, save any remaining text to the last section
        if section_stack:
            current_section_path = self.create_section_path(section_stack)
            if current_section_path not in sections:
                sections[current_section_path] = {'text': []}
            sections[current_section_path]['text'].extend(current_text)
        elif current_text:
            # Add text to the default section if no header was detected
            if default_section not in sections:
                sections[default_section] = {'text': []}
            sections[default_section]['text'].extend(current_text)

        return sections

    def create_section_path(self, section_stack: List[tuple]) -> str:
        """
        Create a hierarchical path string in the format {level 1: xxxx, level 2: xxxx}.
        """
        return ', '.join([f"{title}" for title in section_stack])

# Initialize the parser instance
hierarchical_parser = HierarchicalNodeParser()

@pandas_udf(chunk_schema)
def read_as_hierarchical_chunk(content_series: pd.Series, path_series: pd.Series) -> pd.Series:
    def process_content(content: bytes, path: str) -> List[Dict[str, str]]:
        try:
            document_name = path.replace("dbfs:" + dataset_location, "")  # Strip the dbfs:/ prefix
            elements = parse_pdf(content)
            if not elements:
                return []
            return hierarchical_parser.get_nodes_from_elements(elements, document_name, headers)
        except Exception as e:
            print(f"Error processing content: {str(e)}")
            return []

    return pd.Series([process_content(content, path) for content, path in zip(content_series, path_series)])

# Assuming you have a DataFrame 'df' with columns 'path' and 'content' (content is binary PDF data)
df_chunks = (
    df
    .withColumn("chunk", explode(read_as_hierarchical_chunk(col("content"), col("path"))))
    .select(
        col("path").alias("filepath"),
        col("chunk.text").alias("content"),
        col("chunk.section_header"),
        col("chunk.document_name"), 
        col("chunk.chunk_id"),
        col("chunk.token_count")
    )
)

# Display the resulting DataFrame with parsed content
display(df_chunks)

# COMMAND ----------


# Add a URL column with the base URL for the SCDF website and the corresponding section header appended to the back
from pyspark.sql.functions import col, when, regexp_extract, lit, concat

scdf_base_url = "https://www.scdf.gov.sg/firecode/table-of-content/chapter-8-emergency-lighting-voice-communicarion-systems/clause-"

# Apply the extraction and URL creation in one step
df_chunks = df_chunks.withColumn(
    "url", 
    when(
        regexp_extract(col("section_header"), r'^(8\.\d+)', 1) != "", 
        concat(lit(scdf_base_url), regexp_extract(col("section_header"), r'^(8\.\d+)', 1).cast("string"))
    ).otherwise(None)
)

display(df_chunks)

# COMMAND ----------

# create the sentence transformer (can skip if model exist)

import mlflow
import mlflow.pyfunc
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Load the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Use your model name here

    def predict(self, context, model_input):
        def encode_text(text):
            try:
                return self.model.encode(text, show_progress_bar=False)
            except Exception as e:
                print(f"Error encoding text: {text}. Error: {str(e)}")
                return np.zeros(384)  # Assuming embedding size is 384

        # Apply the model to the input DataFrame
        text_series = model_input.iloc[:, 0]  # Assuming the first column is the text
        embeddings = text_series.apply(encode_text)
        return pd.DataFrame(embeddings.tolist())

# Create sample input and output
sample_input = pd.DataFrame({'text': ["This is a sample sentence"]})
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create sample output by encoding the text
sample_output = pd.DataFrame([model.encode("This is a sample sentence").tolist()])

# Infer the model signature
from mlflow.models.signature import infer_signature
signature = infer_signature(sample_input, sample_output)

run_name="all-MiniLM-L6-v2-run"

# Log the model to MLflow
with mlflow.start_run(run_name=run_name) as run:
    mlflow.pyfunc.log_model(
        artifact_path="hugging_face_sentence_transformer_model",
        python_model=SentenceTransformerModel(),
        input_example=sample_input,
        signature=signature,
        registered_model_name="hugging_face_sentence_transformer_model"  # Specify the name in the registry
    )

# COMMAND ----------

# Check the registered models
import mlflow

print(mlflow.__version__)

from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Search for all registered models
registered_models = client.search_registered_models()
embedding_model = ""
embedding_model_name = ""
embedding_model_full_name = ""
embedding_model_version = 1

# Print details of each registered model
for model in registered_models:
    if (f"{catalog_name}.{schema_name}") in model.name:

        embedding_model_full_name = model.name
        embedding_model_name = model.name.replace(f"{catalog_name}.{schema_name}.", "")
        print(f"Name: {embedding_model_name}")

        embedding_model = client.get_registered_model(name=model.name)
        versions = client.search_model_versions(filter_string=f"name='{model.name}'")

        for version in versions:
            print(f" - Version: {version.version}, Stage: {version.current_stage}")
        
        # Get the latest version    
        embedding_model_version= versions[-1].version

run_id = client.get_model_version(name=embedding_model_full_name, version=embedding_model_version).run_id

try:
    loaded_embedding_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{embedding_model_name}")
    print(f"Model {embedding_model_name} loaded successfully from run ID")
except Exception as e:
    print(f"Error loading from run ID: {str(e)}")

# COMMAND ----------

# Generate all the embeddings
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
from pyspark.sql import Row
import mlflow.pyfunc
import pandas as pd

# Define the schema for the resulting DataFrame
schema = StructType([
    StructField("filepath", StringType(), True),
    StructField("content", StringType(), True),
    StructField("section", StringType(), True), 
    StructField("document_name", StringType(), True),
    StructField("chunk_id", IntegerType(), True),
    StructField("token_count", IntegerType(), True),
    StructField("url", StringType(), True), 
    StructField("content_embedding", ArrayType(FloatType()), True)
])

# Function to generate embeddings for a single row
def generate_embeddings_for_row(row):
    # Get content and section header from the row
    content_input = {'text': [row['content']]}
    section_header_input = {'text': [row['section_header']]}

   # Generate embeddings
    content_embedding = loaded_embedding_model.predict(pd.DataFrame(content_input)).to_numpy().flatten().tolist()  # Convert to list of floats

    # Return a new Row with embeddings and all fields as per the schema
    return Row(
        filepath=row['filepath'],
        content=row['content'],
        section=row['section_header'],  
        document_name=row['document_name'],
        chunk_id=row['chunk_id'],
        token_count=row['token_count'],
        url=row['url'],
        content_embedding=content_embedding
    )

# Apply the function to each row and collect results into a new list of rows
rows_with_embeddings = [generate_embeddings_for_row(row) for row in df_chunks.collect()]

# Convert the list of rows with embeddings back into a Spark DataFrame with the explicit schema
df_with_embeddings = spark.createDataFrame(rows_with_embeddings, schema=schema)

# Display the resulting DataFrame
display(df_with_embeddings)


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the embedding chunk delta table
# MAGIC DROP TABLE IF EXISTS code_regulations_engineering_chunk_embedding;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS code_regulations_engineering_chunk_embedding(
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   filepath STRING,
# MAGIC   content STRING,
# MAGIC   section STRING,
# MAGIC   document_name STRING,
# MAGIC   chunk_id INT,
# MAGIC   token_count INT,
# MAGIC   url STRING,
# MAGIC   content_embedding ARRAY<FLOAT>
# MAGIC   -- NOTE: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC

# COMMAND ----------

# Define the table name
embedding_table_name = f"{catalog_name}.{schema_name}.code_regulations_engineering_chunk_embedding"

# Write the DataFrame to the Delta table with both embeddings
df_with_embeddings.write.mode("append").saveAsTable(embedding_table_name)


# COMMAND ----------

# Check if the vector endpoint is ready
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

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
        vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

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


vsc = VectorSearchClient(disable_notice=True)
VECTOR_SEARCH_ENDPOINT_NAME = f"vs_endpoint_{catalog_name}"

if VECTOR_SEARCH_ENDPOINT_NAME not in [e["name"] for e in vsc.list_endpoints()["endpoints"]]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

# check the status of the endpoint
vs_endpoint_name = VECTOR_SEARCH_ENDPOINT_NAME

wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# the table we'd like to index
source_table_fullname = f"{catalog_name}.{schema_name}.code_regulations_engineering_chunk_embedding"

# where we want to store our index
vs_index_fullname = f"{catalog_name}.{schema_name}.code_regulations_engineering_self_managed_vs_index"

# COMMAND ----------

# create or sync the index to the vector endpoint
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=384, #Match your model embedding size (all-MiniLM-L6-v2)
    embedding_vector_column="content_embedding"
  )
else:
  # trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

  # let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC #### RAG Langchain - Code and Regulation

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.embeddings import DatabricksEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.docstore.document import Document
from flashrank import Ranker, RerankRequest

import mlflow.pyfunc
import pandas as pd


def code_regulation_get_retriever():

    def retrieve(query, k: int=10):
        if isinstance(query, dict):
            query = next(iter(query.values()))

        # get the vector search index
        vsc = VectorSearchClient(disable_notice=True)
        vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)
        
        # get the query vector
        query_embeddings = loaded_embedding_model.predict(pd.DataFrame({"text": [query]}))
        query_vector = query_embeddings.iloc[0].tolist()  # Convert DataFrame to list

        # Perform the similarity search and get similar k documents
        return query, vs_index.similarity_search(
            query_vector=query_vector,
            columns=["id", "document_name", "section", "content", "url"],
            num_results=k)
        
    def rerank(query, retrieved, ranker_model, k: int=10):
        # format result to align with reranker lib format 
        passages = []
        for doc in retrieved.get("result", {}).get("data_array", []):
            new_doc = {
                "score": doc[-1], 
                "id" : doc[0], 
                "file": doc[1], 
                "section": doc[2], 
                "chunk": doc[3],
                "url":doc[4]
                }
            passages.append(new_doc)       
        # Load the flashrank ranker
        ranker = Ranker(model_name=ranker_model)

        # rerank the retrieved documents
        rerankrequest = RerankRequest(query=query, passages=
                                      [{'id': passage['id'],
                                        'text': passage['chunk'],
                                        "meta": {
                                            "section": passage['section'],
                                            "document_name": passage['file'],
                                            "url":passage['url']
                                                 }
                                        } for passage in passages]
                                      )
        results = ranker.rerank(rerankrequest)[:k]
      
        # format the results of rerank to be ready for prompt
        return [Document(page_content=r.get("text"),
                         metadata={
                             "rank_score": r.get("score"), 
                             "vector_id": r.get("id"),
                             "section": r.get("meta").get("section"),
                             "document_name": r.get("meta").get("document_name"),
                             "url": r.get("meta").get("url")}) for r in results]

    # the retriever is a runnable sequence of retrieving and reranking.
    return RunnableLambda(retrieve) | RunnableLambda(lambda x: rerank(x[0], x[1],"ms-marco-MiniLM-L-12-v2"))

# COMMAND ----------

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda

code_regulation_prompt_template = """
    You are an assistant specializing in building codes, safety regulations, and design standards for various room types in buildings. Your task is to extract and provide relevant information about codes, regulations, and standards for the room type mentioned in the question.

    Use the following pieces of context and metadata:

    <context>
    {context}
    </context>

    <metadata>
    Section: {metadata_section}
    </metadata>

    Follow these steps:
    1. Identify the room type mentioned in the question.
    2. Extract and list all relevant codes, regulations, standards, and requirements for the identified room type. Include specific measurements, materials, equipment, location, and any other pertinent details if available.
    3. Organize the information clearly, grouping related requirements together.
    4. If specific information for the mentioned room type is not available, provide general building codes or regulations that might be applicable.

    Provide only factual information from the given context. Do not make assumptions or assessments about compliance. If certain information is not available, clearly state this.

    Question: {input}

    Answer:
    """

# Unwrap the LangChain document from the context to be a dict for logging and MLflow purposes

def unwrap_code_regulation_query_document(answer):
    """
    In answer has the following:
        context
        metadata_section
        input
        input_documents
        output_text
    """
    return answer | {"context": [{"metadata": answer.get("metadata_section"), "page_content": answer.get("context")}]}

def code_regulation_process_question(question, chain):
    # Step 1: Retrieve relevant documents based on the question
    vectorstore = code_regulation_get_retriever()
    similar_documents = vectorstore.invoke(question)

    # Step 2: Group documents by metadata (document name and section) and gather content + metadata in one loop
    grouped_documents = {}
    compiled_context = []
    metadata_section = []
    
    for doc in similar_documents:
        key = f"{doc.metadata.get('document_name')} - {doc.metadata.get('section')}"
        
        # Group document content by metadata
        if key not in grouped_documents:
            grouped_documents[key] = []
        
        grouped_documents[key].append(doc.page_content)
        
        # Check for URL and append it to metadata info
        if key not in metadata_section:
            metadata_info = f"Section: {key}"
            url = doc.metadata.get('url')
            if url:
                metadata_info += f", URL: {url}"
            metadata_section.append(metadata_info)

    # Step 3: Compile the context and metadata only once, minimizing string concatenation in the loop
    final_context = "\n\n".join("\n".join(pages) for pages in grouped_documents.values())
    final_metadata_section = ", ".join(metadata_section)

    # Step 4: Prepare the input for the chain
    input_data = {
        "context": final_context,  # Combined content for all documents
        "metadata_section": final_metadata_section,  # Combined metadata
        "input": question["input"],  # The user's input question
        "input_documents": similar_documents  # The list of documents for StuffDocumentsChain
    }

    # Step 5: Invoke the final chain once with the input data
    answer = chain.invoke(input_data)

    # Step 6: Return the final compiled answer
    return answer.get("output_text")

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

# Create the document prompt for each document
code_regulation_query_document_prompt = PromptTemplate(
    input_variables=["page_content"], 
    template="{page_content}"
)

# Define the document variable name
code_regulation_query_document_variable_name = "context"

# Create the main prompt using ChatPromptTemplate
code_regulation_query_prompt = ChatPromptTemplate.from_template(code_regulation_query_template)

# Create the LLM chain for processing
code_regulation_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens = 3000)
code_regulation_query_llm_chain = LLMChain(llm=code_regulation_model, prompt=code_regulation_query_prompt)

# Create the StuffDocumentsChain with the LLM chain and document prompt
code_regulation_chain = StuffDocumentsChain(
    llm_chain=code_regulation_query_llm_chain,
    document_prompt=code_regulation_query_document_prompt,
    document_variable_name=code_regulation_query_document_variable_name,
)

# Combine the chains and the unwrapping function
code_regulation_final_chain = code_regulation_chain | RunnableLambda(unwrap_code_regulation_query_document)

# COMMAND ----------

# Example question to pass to your process_question function
code_regulation_question_1 =  {"input": "What are the regulations for fire exits?"}

# Call the function to get the answer
code_regulation_answer_1 = code_regulation_process_question(code_regulation_question_1, code_regulation_final_chain)

print(code_regulation_answer_1)

# COMMAND ----------

# Example question to pass to your process_question function
code_regulation_question_1 =  {"input": "What are the regulations for fcc?"}

# Call the function to get the answer
code_regulation_answer_1 = code_regulation_process_question(code_regulation_question_1, code_regulation_final_chain)

print(code_regulation_answer_1)

# COMMAND ----------

# Only run this if want to create model in model registry
from mlflow.models import infer_signature
import mlflow
import langchain

code_regulation_model_name = "code_regulations_rag_model"

# save the langchain model to registry if needed
with mlflow.start_run(run_name="code_regulations_rag_model") as run:
    # Infer the signature from the question and answer
    code_regulation_signature = infer_signature(code_regulation_question, code_regulation_answer)

    # Log the code_regulation_final_chain model to MLflow with retriever
    model_info = mlflow.langchain.log_model(
        code_regulation_final_chain,
        loader_fn=code_regulation_get_retriever,  # Ensure get_retriever is defined
        artifact_path="code_regulation_final_chain",
        registered_model_name=code_regulation_model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=code_regulation_question,  # Set the input example as the question
        signature=code_regulation_signature  # Add the inferred signature
    )
