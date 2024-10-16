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
# MAGIC #### Query Intent Chain

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatDatabricks
import langchain
import pandas as pd

# Initialize the LLM
query_intent_category_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=500)

# Updated categories
query_intent_categories = [
    "Building Codes and Regulations",
    "BIM Revit Data",
    "Compliance Check",
    "Other"
]

# Intent query template
query_intent_category_template = """
You are an AI assistant tasked with categorizing user queries related to building codes, regulations, architectural data, and BIM (Building Information Modeling) elements from Revit. 
Given the following categories:

{categories}

Classify the following query into one of these categories. If the query doesn't fit any category, classify it as "Other".
Use the following guidelines:

1. "Building Codes and Regulations": Queries about specific building codes, regulations, standards, room types (e.g., ELV rooms), disciplines (e.g., ELV, Electrical, Mechanical), or regulatory requirements. This includes questions about which rooms belong to or are managed by specific disciplines.

2. "BIM Revit Data": Queries about physical characteristics of the building such as room id, sizes, locations, boundaries, room relationships, adjacencies, or counts of generic room types. This includes any spatial or structural data typically found in a Revit model. It does not include any information about the discipline that owns or manages which room, nor any regulatory or standard-based information.

3. "Compliance Check": Queries that explicitly ask about how or whether the room complies with regulations or standards.

4. "Other": Queries that don't fit into the above categories.

Respond with only the category name, nothing else.

User Query: {query}

Category:"""

query_intent_category_prompt = PromptTemplate(
    input_variables=["categories", "query"],
    template=query_intent_category_template
)

# Create the classification chain
query_intent_category_classification_chain = LLMChain(llm=query_intent_category_model, prompt=query_intent_category_prompt)

# Define a wrapper function for the model
def classify_query_intent_category(query):
    return query_intent_category_classification_chain.run(categories="\n".join(query_intent_categories), query=query)

# Save the LangChain model to registry
query_intent_category_model_name = "query_intent_category_model"

# Example input for signature inference (as pandas DataFrame)
query_intent_category_example_input = pd.DataFrame({"query": ["What are the building codes for ELV rooms?"]})
query_intent_category_example_output = classify_query_intent_category(query_intent_category_example_input["query"][0])

print(f"Example input: {query_intent_category_example_input['query'][0]}")
print(f"Example output: {query_intent_category_example_output}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Query Intent Model

# COMMAND ----------

# Create a custom MLflow PythonModel class
class QueryIntentCategoryPythonModel(mlflow.pyfunc.PythonModel):
    def __init__(self, categories, template):
        self.categories = categories
        self.template = template

    def load_context(self, context):
        from langchain.chains import LLMChain
        from langchain_community.chat_models import ChatDatabricks
        from langchain.prompts import PromptTemplate

        self.llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=500)
        self.prompt = PromptTemplate(
            input_variables=["categories", "query"],
            template=self.template
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def predict(self, context, model_input):
        query = model_input["query"][0]
        return self.chain.run(categories="\n".join(self.categories), query=query)

# COMMAND ----------

# Start an MLflow run and log the query intent model
with mlflow.start_run(run_name="query_intent_category_model") as run:
    # Infer the signature from the example input and output
    query_intent_category_signature = infer_signature(query_intent_category_example_input, query_intent_category_example_output)

    # Log the model to MLflow
    query_intent_category_model_info = mlflow.pyfunc.log_model(
        artifact_path=query_intent_category_model_name,
        python_model=QueryIntentCategoryPythonModel(query_intent_categories, query_intent_category_template),
        artifacts={},
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"langchain=={langchain.__version__}",
            "pandas"
        ],
        input_example=query_intent_category_example_input,
        signature=query_intent_category_signature,
        registered_model_name=query_intent_category_model_name
    )

print(f"Model '{query_intent_category_model_name}' saved successfully to the MLflow Model Registry!")
print(f"Model URI: {query_intent_category_model_info.model_uri}")

# COMMAND ----------

# Check the registered models
import mlflow

print(mlflow.__version__)

from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Search for all registered models
registered_models = client.search_registered_models()
query_intent_model = ""
query_intent_model_name = ""
query_intent_model_full_name = ""
query_intent_model_version = 1

# Print details of each registered model
for model in registered_models:
    if (f"{catalog_name}.{schema_name}.{query_intent_category_model_name}") in model.name:
        
        query_intent_model_full_name = model.name
        query_intent_model_name = model.name.replace(f"{catalog_name}.{schema_name}.", "")
        print(f"Name: {query_intent_model_name}")

        query_intent_model = client.get_registered_model(name=model.name)
        versions = client.search_model_versions(filter_string=f"name='{model.name}'")

        for version in versions:
            print(f" - Version: {version.version}, Stage: {version.current_stage}")
        
        # Get the latest version    
        query_intent_model_version= versions[-1].version

print(f"Model: {query_intent_model_full_name}")
run_id = client.get_model_version(name=query_intent_model_full_name, version=query_intent_model_version).run_id

try:
    loaded_query_intent_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{query_intent_model_name}")
    print(f"Model {query_intent_model_name} loaded successfully from run ID")
except Exception as e:
    print(f"Error loading from run ID: {str(e)}")


# COMMAND ----------

from rich.console import Console
from rich.markdown import Markdown

query_intent_category_question = {"query": ["What are FCC Room Requirements I have to comply with?"]}
query_intent_category_answer = loaded_query_intent_model.predict(query_intent_category_question)
console = Console()
console.print(Markdown(query_intent_category_answer))
