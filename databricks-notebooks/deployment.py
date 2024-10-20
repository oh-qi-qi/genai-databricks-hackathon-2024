# Databricks notebook source
import subprocess
import sys

# List of packages to check and install if not present
packages = {
    "mlflow": "2.16.2",
    "transformers": "4.36.1",
    "llama-index": "0.11.17",
    "pydantic": "2.7.4",
    "accelerate": "0.25.0",
    "PyPDF2": "3.0.1",
    "PyMuPDF": "1.24.11",
    "sentencepiece": "0.1.99",
    "pdfminer.six": "20240706",
    "sentence-transformers": "2.2.2",
    "databricks-vectorsearch": "0.42",
    "langchain": "0.3.3",
    "langchain-community": "0.3.2",
    "flashrank": "0.2.9",
    "rich": "13.9.2",
    "fuzzywuzzy": "0.18.0",
    "python-Levenshtein": "0.26.0",
    "inflection": "0.5.1",
    "cloudpickle": "2.0.0",
    "langchain-core": "0.3.10",
    "langsmith": "0.1.132",
    "databricks-sdk": "0.34.0",
    "networkx": "3.4.1"
}

def install_package(package, version=""):
    if version:
        package = f"{package}=={version}"
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package, version in packages.items():
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing...")
        install_package(package, version)

print("All packages are checked and installed if necessary.")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Compliance Chain

# COMMAND ----------

catalog_name = "llm_workspace"
schema_name = "default"
volume_name = "regubim-ai-volume"
dataset_location = f"Volumes/{catalog_name}/{schema_name}/{volume_name}/"


# COMMAND ----------

# Standard library imports
import os
import re
import ast
import json
import logging
from typing import List, Dict, Any, Union

import logging
# Configure logging for PySpark
logging.getLogger("pyspark").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)

# If you're using PySpark SQL specifically, you might also want to add:
logging.getLogger("pyspark.sql.connect.client.logging").setLevel(logging.ERROR)

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx
import inflection
from fuzzywuzzy import process  # FuzzyWuzzy (for string matching)

# PySpark and GraphFrames
from pyspark.sql import SparkSession

# LangChain imports
from langchain_community.chat_models import ChatDatabricks
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.schema.runnable import RunnableMap, RunnableBranch, RunnablePassthrough
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser

# Rich (for console output)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# IPython display
from IPython.display import HTML, display

# Databricks specific imports
from databricks.vector_search.client import VectorSearchClient

# FlashRank imports
from flashrank import Ranker, RerankRequest

# MLflow imports
import mlflow
import mlflow.pyfunc
import cloudpickle
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient

# COMMAND ----------

class MultiStageSystemWrapper():
    def __init__(self, llm_model_name, catalog_name, schema_name, volume_name):
        self.llm_model_name = llm_model_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.volume_name = volume_name
        self.DEBUG_MODE = False  # Default value
    
    def debug_print(self, message):
        if self.DEBUG_MODE:
            print(message)

    @staticmethod
    def create_chain_query_classification(llm_model):
        # Query intent categories
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
        inner_chain = (
            RunnablePassthrough.assign(categories=lambda _: "\n".join(query_intent_categories))
            | query_intent_category_prompt
            | llm_model
            | StrOutputParser()
        )

        # Wrapper function to include input and output
        def chain_with_io(inputs):
            result = inner_chain.invoke(inputs)
            return {
                'input': inputs['query'],
                'output': result
            }

        # Convert the wrapper function to a RunnableLambda
        return RunnableLambda(chain_with_io)

    @staticmethod
    def create_code_regulation_rag_chain(llm_model):
        vs_endpoint_name = f"vs_endpoint_{catalog_name}"
        vs_index_fullname = f"{catalog_name}.{schema_name}.code_regulations_engineering_self_managed_vs_index"

        # get embedding model
        client = MlflowClient()

        # Search for all registered models
        registered_models = client.search_registered_models()
        embedding_model = ""
        embedding_model_name = "hugging_face_sentence_transformer_model"
        embedding_model_full_name = ""
        embedding_model_version = 1

        # Print details of each registered model
        for model in registered_models:
            if (f"{catalog_name}.{schema_name}.{embedding_model_name}") in model.name:

                embedding_model_full_name = model.name
                embedding_model = client.get_registered_model(name=model.name)
                versions = client.search_model_versions(filter_string=f"name='{model.name}'")

                for version in versions:
                    print(f" - Version: {version.version}, Stage: {version.current_stage}")
                
                # Get the latest version    
                embedding_model_version= versions[0].version

        run_id = client.get_model_version(name=embedding_model_full_name, version=embedding_model_version).run_id

        try:
            loaded_embedding_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{embedding_model_name}")
            print(f"Model {embedding_model_name} loaded successfully from run ID {run_id}")
        except Exception as e:
            print(f"Error loading from run ID: {str(e)}")

        def retrieve(inputs):
            query = inputs
            vsc = VectorSearchClient(disable_notice=True)
            vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)
            query_embeddings = loaded_embedding_model.predict(pd.DataFrame({"text": [query]}))
            query_vector = query_embeddings.iloc[0].tolist()
            retrieved = vs_index.similarity_search(
                query_vector=query_vector,
                columns=["id", "document_name", "section", "content", "url"],
                num_results=10)
            return {'query': query, 'retrieved': retrieved}

        def rerank(inputs, ranker_model="ms-marco-MiniLM-L-12-v2", k=10):
            query = inputs['query']
            retrieved = inputs['retrieved']
            passages = [
                {
                    "score": doc[-1], 
                    "id": doc[0], 
                    "file": doc[1], 
                    "section": doc[2], 
                    "chunk": doc[3],
                    "url": doc[4]
                } for doc in retrieved.get("result", {}).get("data_array", [])
            ]
            ranker = Ranker(model_name=ranker_model)
            rerankrequest = RerankRequest(query=query, passages=[
                {
                    'id': passage['id'],
                    'text': passage['chunk'],
                    "meta": {
                        "section": passage['section'],
                        "document_name": passage['file'],
                        "url": passage['url']
                    }
                } for passage in passages
            ])
            results = ranker.rerank(rerankrequest)[:k]
            reranked_docs = [Document(
                page_content=r.get("text"),
                metadata={
                    "rank_score": r.get("score"), 
                    "vector_id": r.get("id"),
                    "section": r.get("meta").get("section"),
                    "document_name": r.get("meta").get("document_name"),
                    "url": r.get("meta").get("url")
                }
            ) for r in results]
            return {'query': query, 'retrieved_docs': reranked_docs}

        def process_retrieved_docs(inputs):
            query = inputs['query']
            docs = inputs['retrieved_docs']
            grouped_documents = {}
            metadata_section = []
            
            for doc in docs:
                key = f"{doc.metadata.get('document_name')} - {doc.metadata.get('section')}"
                if key not in grouped_documents:
                    grouped_documents[key] = []
                grouped_documents[key].append(doc.page_content)
                
                if key not in metadata_section:
                    metadata_info = f"Section: {key}"
                    url = doc.metadata.get('url')
                    if url:
                        metadata_info += f", URL: {url}"
                    metadata_section.append(metadata_info)

            final_context = "\n\n".join("\n".join(pages) for pages in grouped_documents.values())
            final_metadata_section = ", ".join(metadata_section)

            return {
                "context": final_context,
                "metadata_section": final_metadata_section,
                "input": query,
                "input_documents": docs
            }

        def format_output(inputs):
            output_text = inputs.get('output_text', '')
            input_documents = inputs.get('input_documents', [])
            
            references_dict = {}
            
            for doc in input_documents:
                doc_name = doc.metadata.get('document_name', 'Unknown Document')
                full_section = doc.metadata.get('section', 'Unknown Section')
                url = doc.metadata.get('url', '')
                
                parts = full_section.split(',', 1)
                main_section = parts[0].strip()
                subsection = parts[1].strip() if len(parts) > 1 else ''
                
                if doc_name not in references_dict:
                    references_dict[doc_name] = {'url': url, 'main_section': main_section, 'subsections': set()}
                
                if subsection:
                    references_dict[doc_name]['subsections'].add(subsection)
            
            references = []
            for doc_name, info in references_dict.items():
                url = info['url']
                main_section = info['main_section']
                subsections = ', '.join(sorted(info['subsections']))
                
                if subsections:
                    section_text = f"{main_section}, {subsections}"
                else:
                    section_text = main_section
                
                if url:
                    reference = f"* [{doc_name}]({url}), Section: {section_text}"
                else:
                    reference = f"* {doc_name}, Section: {section_text}"
                
                references.append(reference)
            
            if references:
                references_text = "\n".join(sorted(references))
                output_text += f"\n\nTo ensure compliance, please refer to the following documents:\n\n{references_text}"
            
            return {'input': inputs.get('input'), 'output': output_text}
        
        # Define the main prompt template
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

        # Create the main prompt
        code_regulation_prompt = ChatPromptTemplate.from_template(code_regulation_prompt_template)

        # Create the document prompt
        code_regulation_prompt_document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # Create the LLM chain
        llm_chain = LLMChain(llm=llm_model, prompt=code_regulation_prompt)

        # Create the StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=code_regulation_prompt_document_prompt,
            document_variable_name="context"
        )

        # Combine the chains
        retriever_chain = RunnableLambda(retrieve) | RunnableLambda(rerank)
        final_chain = retriever_chain | RunnableLambda(process_retrieved_docs) | stuff_chain | RunnableLambda(format_output)

        return final_chain

    @staticmethod
    def create_room_identification_chain(llm_model):
        room_identification_prompt = """
        Input Query: {input}

        Extracted Regulatory Information:
        {code-regulation-rag-chain-output}

        Task:
        Analyze the input query and extracted regulatory information to identify rooms and their relationships. Then, generate a focused query to check room existence and connectivity.

        Steps:
        1. Identify the primary room(s) explicitly mentioned or strongly implied in the input query.
        2. Determine related rooms based on regulatory requirements and spatial relationships.
        3. For any `or`, `and` statements in room names, list each room option separately.
        4. Formulate a simple query to check the existence of identified rooms and the paths between them.

        Output Format:
        Primary Room(s): [Primary Room Name(s)]
        Related Rooms:
        - [Related Room 1]: [Brief explanation of relationship or requirement]
        - [Related Room 2a] or [Related Room 2b]: [Brief explanation of relationship or requirement]
        - [Related Room 3]: [Brief explanation of relationship or requirement]
        ...

        Generated Query:
        Check if the following rooms exist: [Primary Room Name(s)], [Related Room 1], [Related Room 2a], [Related Room 2b], [Related Room 3], ...
        If [Primary Room Name] exists, what are the paths from [Primary Room Name] to [Related Room 1], [Related Room 2a], [Related Room 2b], [Related Room 3], ...? 
        If there are rooms that do not exist, please state in the answer.

        Note: 
        - Only include actual rooms or defined spaces, not general areas.
        - Prioritize relationships and paths explicitly mentioned in the regulatory information or critical for compliance and safety.
        - Ensure all identified rooms are relevant to the input query and regulatory requirements.
        """

        room_identification_prompt_template = ChatPromptTemplate.from_template(room_identification_prompt)
        room_identification_chain = LLMChain(llm=llm_model, prompt=room_identification_prompt_template)

        def invoke(inputs: dict) -> dict:
            chain_output = room_identification_chain.invoke(inputs)
            return {
                'input': inputs['input'],
                'code-regulation-rag-chain-output': inputs['code-regulation-rag-chain-output'],
                'room-identification-chain-output': chain_output['text']
            }

        return invoke
    
    @staticmethod
    def create_bim_revit_data_chain(catalog_name, schema_name, llm_model, volume_name):
        # List of Tools DataRetrievalTool, RoomPathCalculationTool, RoomRelationshipTool, and TableGenerationTool classes 
        class DataRetrievalTool:
            def __init__(self, spark, catalog_name, schema_name):
                self.spark = spark
                self.catalog_name = catalog_name
                self.schema_name = schema_name

            def get_room_vertices(self):
                table_name = f"{self.catalog_name}.{self.schema_name}.revit_room_vertices"
                return self.spark.read.table(table_name)

            def get_room_edges(self):
                table_name = f"{self.catalog_name}.{self.schema_name}.revit_room_edges"
                return self.spark.read.table(table_name)

        class RoomPathCalculationTool:
            def __init__(self, data_retrieval_tool, catalog_name, schema_name, volume_name, template_filename):
                self.data_retrieval_tool = data_retrieval_tool

                # Construct the file path
                template_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{template_filename}"

                # Load the HTML template
                if os.path.exists(template_path):
                    with open(template_path, 'r') as file:
                        self.html_template = file.read()
                else:
                    raise FileNotFoundError(f"Template file not found at {template_path}")

            def calculate(self, input_str):
                try:
                    # Split the input into source_room and target_room
                    source_room, target_room = [item.split(":")[1].strip() for item in input_str.split(",")]

                    room_edges_df_from_spark = self.data_retrieval_tool.get_room_edges()
                    result = self.find_all_shortest_paths(room_edges_df_from_spark, source_room.upper(), target_room.upper())

                    if isinstance(result, str):
                        return result, None  # Error message, no graph JSON
                    else:
                        paths_df, graph_json = result

                        # Load the JSON string into a Python object
                        graph_data = json.loads(graph_json)

                        # Clean up the 'links' part of the JSON: ensure double quotes are formatted correctly
                        for link in graph_data.get('links', []):
                            if 'door_name' in link:
                                # Replace any escaped quotes and manually ensure the formatting is correct
                                link['door_name'] = link['door_name'].replace('\\', '').replace('"', '')

                        # Convert back to JSON string
                        clean_graph_json = json.dumps(graph_data, ensure_ascii=False)

                        return paths_df, clean_graph_json

                except Exception as e:
                    return f"Error in calculate method: {str(e)}", None

            def __call__(self, input_str):
                """
                This method allows the tool to be callable, and splits the input string into two arguments.
                The expected format of `input_str` is 'source_room: <source_room>, target_room: <target_room>'.
                """
                try:
                    # Split the input into source_room and target_room
                    source_room, target_room = [item.split(":")[1].strip() for item in input_str.split(",")]

                    # Call the calculate method with the parsed rooms
                    return self.calculate(source_room.upper(), target_room.upper())

                except Exception as e:
                    return f"Error: {str(e)}"

            def generate_visualization(self, room_graph_data_json):
                room_graph_data_json = room_graph_data_json.replace("\'\n","").replace("\'{\"nodes\":","{\"nodes\":")

                # Safely embed the JSON string using JSON.parse in the HTML
                html_content = self.html_template.replace(
                    "'''path_graph_json'''",
                    room_graph_data_json.strip("'")
                )

                html_object = HTML(html_content)

                # Display the HTML
                # displayHTML(html_content)

                # Return both the HTML content and the HTML object
                return html_content, html_object, room_graph_data_json.strip("'")

            def numpy_to_python(self, obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: self.numpy_to_python(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [self.numpy_to_python(item) for item in obj]
                else:
                    return obj

            def calculate_center_point(self, points):
                try:
                    if isinstance(points, np.ndarray):
                        points = points.tolist()
                    if not isinstance(points, list) or len(points) != 4:
                        return {'X': 0, 'Y': 0, 'Z': 0}
                    x_sum = sum(p['X'] for p in points)
                    y_sum = sum(p['Y'] for p in points)
                    z_sum = sum(p['Z'] for p in points)
                    return {'X': x_sum / 4, 'Y': y_sum / 4, 'Z': z_sum / 4}
                except Exception as e:
                    print(f"Error calculating center point: {e}")
                    print(f"Points data: {points}")
                    return {'X': 0, 'Y': 0, 'Z': 0}

            def manhattan_distance(self, point1, point2):
                return (abs(point1['X'] - point2['X']) + 
                        abs(point1['Y'] - point2['Y']) + 
                        abs(point1['Z'] - point2['Z']))

            def find_matching_rooms(self, room_name, room_names, threshold=80):
                matches = process.extract(room_name, room_names, limit=None)
                return [match for match in matches if match[1] >= threshold]

            def custom_path_finder(self, G, source, target):
                def dfs_paths(current, path):
                    if current == target:
                        yield path
                    for neighbor in G.neighbors(current):
                        if neighbor not in path:
                            if "OUTSIDE" not in neighbor or neighbor in (source, target):
                                yield from dfs_paths(neighbor, path + [neighbor])

                return list(dfs_paths(source, [source]))

            def find_all_shortest_paths(self, room_edges_df_from_spark, source_room, target_room):
                # Convert Spark DataFrame to Pandas DataFrame
                room_edges_df = room_edges_df_from_spark.toPandas()

                # Create a graph of rooms
                G = nx.Graph()

                # Add rooms as nodes and connections as edges
                for _, row in room_edges_df.iterrows():
                    src_name, dst_name = row['src_name'], row['dst_name']
                    door_center = self.calculate_center_point(row['door_bounds'])

                    # Add nodes with all available information
                    for name in [src_name, dst_name]:
                        if not G.has_node(name):
                            G.add_node(name,
                                    id=row['src'] if name == src_name else row['dst'],
                                    level=row['src_level'] if name == src_name else row['dst_level'],
                                    area=row['src_area'] if name == src_name else row['dst_area'],
                                    bounds=self.numpy_to_python(row['src_bounds'] if name == src_name else row['dst_bounds']),
                                    type="Room")

                    # Add edge
                    G.add_edge(src_name, dst_name,
                            src_name=src_name,
                            dst_name=dst_name,
                            src_id=row['src'],
                            dst_id=row['dst'],
                            door_id=row['door_id'],
                            door_name=row['door_name'],
                            door_level=row['door_level'],
                            door_center=door_center,
                            door_bounds=self.numpy_to_python(row['door_bounds']))

                # Get unique room names
                room_names = list(G.nodes())

                # Find matches for source and target with score 90 and above
                source_matches = [match for match in process.extract(source_room, room_names, limit=None) if match[1] >= 90]
                target_matches = [match for match in process.extract(target_room, room_names, limit=None) if match[1] >= 90]

                if not source_matches:
                    return "No source room found with match score 90 or above"
                if not target_matches:
                    return "No target room found with match score 90 or above"

                all_paths = []

                for source_room, source_score in source_matches:
                    for target_room, target_score in target_matches:
                        if source_room == target_room:
                            continue  # Skip if source and target are the same

                        try:
                            # Find all simple paths between source and target
                            simple_paths = self.custom_path_finder(G, source_room, target_room)

                            for path in simple_paths:
                                total_distance = 0
                                door_path = []
                                fuzzy_path = []

                                for i in range(len(path)):
                                    room = path[i]
                                    fuzzy_match = process.extractOne(room, room_names)
                                    fuzzy_path.append(fuzzy_match[0])

                                    if i < len(path) - 1:
                                        room1, room2 = path[i], path[i+1]
                                        edge_data = G[room1][room2]
                                        door_id = edge_data['door_id']
                                        door_center = edge_data['door_center']

                                        if i > 0:  # Calculate distance from previous door to this door
                                            distance = self.manhattan_distance(prev_door_center, door_center)
                                            total_distance += distance

                                        door_path.append((door_id, edge_data['door_name'], edge_data['door_level']))
                                        prev_door_center = door_center

                                all_paths.append((fuzzy_path, total_distance, source_room, target_room, door_path, source_score, target_score))

                        except nx.NetworkXNoPath:
                            continue
                        except Exception as e:
                            print(f"Error processing path: {e}")
                            continue

                if not all_paths:
                    return f"No paths found between any matching source and target rooms"

                # Sort paths by distance
                all_paths.sort(key=lambda x: x[1])

                # Create a DataFrame from all_paths
                paths_df = pd.DataFrame(all_paths[:100], columns=['Path', 'Distance', 'Source', 'Target', 'DoorPath', 'SourceScore', 'TargetScore'])

                # Create graph JSON
                graph_json = {"nodes": [], "links": []}
                unique_rooms = set()
                for path, _, _, _, door_path, _, _ in all_paths:
                    for room in path:
                        if room not in unique_rooms:
                            unique_rooms.add(room)
                            node_data = G.nodes[room]
                            graph_json["nodes"].append({
                                "id": node_data['id'],
                                "name": room,
                                "level": node_data['level'],
                                "area": node_data['area'],
                                "type": node_data['type']
                            })

                for i, (path, total_distance, _, _, door_path, _, _) in enumerate(all_paths):
                    for j in range(len(path) - 1):
                        source, target = path[j], path[j+1]
                        edge_data = G[source][target]

                        # Determine the correct source and target IDs
                        source_id = edge_data['src_id'] if edge_data['src_name'] == source else edge_data['dst_id']
                        target_id = edge_data['dst_id'] if edge_data['dst_name'] == target else edge_data['src_id']

                        graph_json["links"].append({
                            "source": source_id,
                            "target": target_id,
                            "source_name": source,
                            "target_name": target,
                            "door_id": edge_data['door_id'],
                            "door_name": edge_data['door_name'],
                            "door_level": edge_data['door_level'],
                            "route": i + 1,  # Add route number
                            "route_distance": total_distance,  # Add total distance for the entire route
                            "path": path  # Add the entire path as an array
                        })

                # Minified JSON
                return paths_df, json.dumps(self.numpy_to_python(graph_json), ensure_ascii=False, separators=(',', ':'))

        class RoomRelationshipTool:
            def __init__(self, data_retrieval_tool, catalog_name, schema_name, volume_name, template_filename):
                self.data_retrieval_tool = data_retrieval_tool

                # Construct the file path
                template_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{template_filename}"

                # Load the HTML template
                if os.path.exists(template_path):
                    with open(template_path, 'r') as file:
                        self.html_template = file.read()
                else:
                    raise FileNotFoundError(f"Template file not found at {template_path}")

                # Define intents and their associated phrases
                self.intents = {
                    "count_rooms": ["how many rooms", "how many rooms in the building", "count rooms", "total rooms"],
                    "count_connections": ["how many connections", "count connections", "total connections", "total number of connections", "number of connections", "total number of paths",  "how many paths", "total paths"],
                    "list_rooms": ["list rooms", "show rooms", "what rooms", "room list", "list rooms in the building"],
                    "check_rooms_exist": ["check rooms exist", "do these rooms exist", "are these rooms present", "check if the following rooms exist", "check if the rooms exist", "check if these rooms exist"]
                }

            def analyze(self, query=None):
                vertices = self.data_retrieval_tool.get_room_vertices()
                edges = self.data_retrieval_tool.get_room_edges()

                room_graph_data_json = self.create_graph_json(vertices, edges)
                room_graph = self.create_nx_graph(vertices, edges)
                analysis_result = self.perform_analysis(room_graph, query)

                # print(f"Vertices in Spark DataFrame: {vertices.count()}")
                # print(f"Edges in Spark DataFrame: {edges.count()}")
                # print(f"Nodes in NetworkX graph: {room_graph.number_of_nodes()}")
                # print(f"Edges in NetworkX graph: {room_graph.number_of_edges()}")

                return room_graph, room_graph_data_json, analysis_result

            def create_nx_graph(self, vertices, edges):
                G = nx.MultiGraph()

                # Add nodes
                for row in vertices.collect():
                    name = str(row['name'])
                    G.add_node(name, **{k: str(v) for k, v in row.asDict().items()})

                # Create a set of valid room names
                valid_rooms = set(G.nodes())

                # Add edges based on doors
                door_count = 0
                for row in edges.collect():
                    src = str(row['src_name'])
                    dst = str(row['dst_name'])
                    door_id = str(row['door_id'])
                    door_name = str(row['door_name'])

                    if src in valid_rooms and dst in valid_rooms:
                        # Add edge with door information
                        G.add_edge(src, dst, key=door_id, door_name=door_name)
                        door_count += 1
                    else:
                        print(f"Invalid connection: {src} - {dst} (Door: {door_name})")
                return G

            def perform_analysis(self, room_graph, query):
                if not query:
                    return "No specific analysis performed. Please provide a query for detailed analysis."

                intent = self.interpret_query(query)
                if intent == "count_rooms":
                    return f"Total number of rooms: {room_graph.number_of_nodes()}"
                elif intent == "count_connections":
                    return f"Total number of connections: {room_graph.number_of_edges()}"
                elif intent == "list_rooms":
                    room_list = ", ".join(sorted(room_graph.nodes()))
                    return f"List of rooms: {room_list}"
                elif intent == "check_rooms_exist":
                    original_room_names = self.extract_room_names(query)
                    room_names_lower = [name.lower() for name in original_room_names]
                    existing_rooms_lower = [name.lower() for name in room_graph.nodes()]
                
                    existing = []
                    non_existing = []
                    for i, room_lower in enumerate(room_names_lower):
                        best_match = process.extractOne(room_lower, existing_rooms_lower, score_cutoff=80)
                        if best_match:
                            # Find the original (non-lowercased) room name
                            original_name = next(name for name in room_graph.nodes() if name.lower() == best_match[0])
                            existing.append(original_name)
                        else:
                            non_existing.append(original_room_names[i])
                    
                    response = "Here's what I found:\n"
                    if existing:
                        response += f"Existing rooms: {', '.join(existing)}\n"
                    if non_existing:
                        response += f"Rooms not found: {', '.join(non_existing)}\n"
                    if not existing and not non_existing:
                        response += "No rooms were identified in your query."
                    
                    return response
                else:
                    return f"I'm not sure how to analyze '{query}'. Could you please rephrase or provide more details?"


            def interpret_query(self, query):
                query = query.lower()
                all_phrases = [phrase for phrases in self.intents.values() for phrase in phrases]
                best_match = process.extractOne(query, all_phrases)

                if best_match[1] >= 70:
                    matched_phrase = best_match[0]
                    for intent, phrases in self.intents.items():
                        if matched_phrase in phrases:
                            return intent

                if any(word in query for word in ["check", "exist", "present"]):
                    return "check_rooms_exist"

                return "unknown"

            def extract_room_names(self, query):
                query = query.lower()
                patterns = [
                    r"check if the following rooms exist:\s*(.*)",
                    r"check if these rooms exist:\s*(.*)",
                    r"check.*?(?:if|whether).*?((?:[\w\s()]+,\s*)*[\w\s()]+).*?exist",
                    r"do.*?((?:[\w\s()]+,\s*)*[\w\s()]+).*?exist",
                    r"are.*?((?:[\w\s()]+,\s*)*[\w\s()]+).*?present"
                ]

                for pattern in patterns:
                    match = re.search(pattern, query)
                    if match:
                        rooms_text = match.group(1)
                        rooms = re.split(r',\s*|\s+and\s+|\s+or\s+', rooms_text)
                        return [self.clean_room_name(room) for room in rooms if room]

                # If no pattern matched, extract all capitalized words as potential room names
                return [self.clean_room_name(word) for word in re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', query)]

            def clean_room_name(self, room):
                room = room.strip()
                room = re.sub(r'^\(|\)$', '', room)
                return ' '.join(word.capitalize() for word in room.split())

            def generate_visualization(self, room_graph_data_json):
                # Safely embed the JSON string using JSON.parse in the HTML
                html_content = self.html_template.replace(
                    "'''room_graph_data_json'''",
                    room_graph_data_json.replace("\'\n","").replace("\'{\"nodes\":","{\"nodes\":")
                )

                html_object = HTML(html_content)

                # Display the HTML
                # display(html_object)

                # Return both the HTML content and the HTML object
                return html_content, html_object

            def create_graph_json(self, vertices, edges):
                def convert_value(value):
                    if isinstance(value, np.ndarray):
                        return value.tolist()
                    if isinstance(value, np.generic):
                        return value.item()
                    return value

                nodes = [
                    {
                        "id": convert_value(getattr(row, 'id', None)),
                        "name": convert_value(getattr(row, 'name', None)),
                        "level": convert_value(getattr(row, 'level', None)),
                        "area": convert_value(getattr(row, 'area', None)),
                        "type": convert_value(getattr(row, 'type', None))
                    }
                    for row in vertices.toPandas().itertuples()
                ]

                links = [
                    {
                        "source_room_id": convert_value(getattr(row, 'src', None)),
                        "source_room_name": convert_value(getattr(row, 'src_name', None)),
                        "target_room_id": convert_value(getattr(row, 'dst', None)),
                        "target_room_name": convert_value(getattr(row, 'dst_name', None)),
                        "door_id": convert_value(getattr(row, 'door_id', None)),
                        "door_level": convert_value(getattr(row, 'door_level', None))
                    }
                    for row in edges.toPandas().itertuples()
                ]

                room_graph_data = {
                    "nodes": nodes,
                    "links": links
                }

                # Minified JSON
                return json.dumps(room_graph_data, ensure_ascii=False, separators=(',', ':'))

        class TableGenerationTool:
            def __init__(self):
                self.max_rows = 10
                self.max_columns_to_show = 10

            def generate_markdown_table(self, input_data: str) -> str:
                """
                Generate a Markdown table from input JSON data.
                """
                try:
                    data = self._parse_input(input_data)
                    if not data:
                        return "Error: Invalid or empty data"

                    total_items = len(data)
                    headers = self._get_headers(data)

                    # Extract first 10 rows and last row if data is large
                    if total_items > self.max_rows + 1:
                        displayed_data = data[:self.max_rows] + [data[-1]]
                        ellipsis_needed = True
                    else:
                        displayed_data = data
                        ellipsis_needed = False

                    table = self._create_table_header(headers)
                    table += self._create_table_rows(displayed_data, headers, ellipsis_needed)
                    table += self._add_table_footer(total_items, len(headers))

                    return table
                except json.JSONDecodeError as e:
                    return f"Error parsing JSON: {str(e)}\nInput data: {input_data[:100]}..."
                except Exception as e:
                    return f"Error generating table: {str(e)}\nInput data: {input_data[:100]}..."

            def _parse_input(self, input_data: str) -> List[Dict[str, Any]]:
                """Parse the input string as JSON."""
                return json.loads(input_data)

            def _get_headers(self, data: List[Dict[str, Any]]) -> List[str]:
                """Extract unique headers from all items in the data."""
                headers = set()
                for item in data:
                    headers.update(item.keys())
                return sorted(list(headers))

            def _create_table_header(self, headers: List[str]) -> str:
                """Create the Markdown table header."""
                visible_headers = headers[:self.max_columns_to_show]
                if len(headers) > self.max_columns_to_show:
                    visible_headers.append("...")
                header_row = "| " + " | ".join([inflection.titleize(header.replace("_", " ").replace("source", "Start").replace("target", "Destination")) for header in visible_headers]) + " |\n"

                separator_row = "|" + "|".join(["---" for _ in visible_headers]) + "|\n"
                return header_row + separator_row

            def _create_table_rows(self, data: List[Dict[str, Any]], headers: List[str], ellipsis_needed: bool) -> str:
                """Create the Markdown table rows."""
                rows = ""
                total_rows = len(data)
                for idx, item in enumerate(data):
                    if ellipsis_needed and idx == self.max_rows:
                        # Insert ellipsis row
                        row_data = ["..." for _ in headers[:self.max_columns_to_show]]
                        if len(headers) > self.max_columns_to_show:
                            row_data.append("...")
                        row = "| " + " | ".join(row_data) + " |\n"
                        rows += row
                        continue
                    row_data = [str(item.get(header, "")) for header in headers[:self.max_columns_to_show]]
                    if len(headers) > self.max_columns_to_show:
                        row_data.append("...")
                    row = "| " + " | ".join(row_data) + " |\n"
                    rows += row
                return rows

            def _add_table_footer(self, total_items: int, total_columns: int) -> str:
                """Add a footer with information about the number of items and columns."""
                footer = f"\n*Table {'truncated' if total_items > self.max_rows + 1 else 'complete'}. "
                footer += f"Showing {min(self.max_rows + 1, total_items)} out of {total_items} total records. "
                if total_columns > self.max_columns_to_show:
                    footer += f"Displaying {self.max_columns_to_show} out of {total_columns} columns.*"
                else:
                    footer += f"All {total_columns} columns displayed.*"
                return footer

        class BIMRevitDataChain:
            def __init__(self, catalog_name, schema_name,llm_model, volume_name):
                self.catalog_name = catalog_name
                self.schema_name = schema_name
                self.volume_name = volume_name
                self.bim_revit_data_model = llm_model
                self.setup_tools()
                self.agent = None  # We'll set this up later

            def setup_tools(self):
                spark_room_analysis = SparkSession.builder.appName("RoomAnalysis").getOrCreate()
                self.data_retrieval_tool = DataRetrievalTool(spark_room_analysis, self.catalog_name, self.schema_name)
                self.path_calculation_tool = RoomPathCalculationTool(
                    self.data_retrieval_tool, self.catalog_name, self.schema_name, self.volume_name, "room-route-visualisation-min.html"
                )
                self.room_relationship_tool = RoomRelationshipTool(
                    self.data_retrieval_tool, self.catalog_name, self.schema_name, self.volume_name, "room-relationship-visualisation-min.html"
                )
                self.table_generation_tool = TableGenerationTool()

                self.tools = [
                    Tool(
                        name="RoomPathCalculation",
                        func=self.path_calculation_tool.calculate,
                        description="Calculates the path between two rooms. Input must be in the format 'source_room: <SOURCE_ROOM>, target_room: <TARGET_ROOM>'. Room names should be in uppercase."
                    ),
                    Tool(
                        name="RoomPathVisualization",
                        func=self.path_calculation_tool.generate_visualization,
                        description="Generates and returns a visualization of the path between rooms. Input should be the graph JSON returned by RoomPathCalculation.  This tool will automatically display the visualization in the notebook. Returns the visualization JSON."
                    ),
                    Tool(
                        name="RoomRelationshipAnalysis",
                        func=self.room_relationship_tool.analyze,
                        description="Analyzes relationships between rooms. Can accept queries for specific analyses."
                    ),
                    Tool(
                        name="GenerateMarkdownTable",
                        func=self.table_generation_tool.generate_markdown_table,
                        description="Generates a Markdown table from JSON data. Input should be a JSON string or a list of dictionaries."
                    )
                ]

            def setup_agent(self, query=None):
                intent_category = query.get('intent_category', 'Unknown') if query else 'Unknown'
                tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                tool_names = ", ".join([tool.name for tool in self.tools])

                bim_revit_agent_template = f"""
                As an expert in room analysis, answer the following questions to the best of your ability. You have access to the following tools:

                {tool_descriptions}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do and which tool (if any) to use
                Action: the action to take, should be one of [{tool_names}], or 'None' if no action is needed
                Action Input: the input to the action (if any), or 'None' if no action is needed
                Observation: the result of the action (summarize large JSON outputs or split into manageable chunks), or 'None' if no action was taken
                ... (this Thought/Action/Action Input/Observation can repeat N times as needed)
                Thought: I now know the final answer
                Visualization JSON: the JSON data for visualization, or 'None' if not applicable
                Final Answer:  the final answer to the question and include any relevant note as the value and the key is `final_answer` and the Visualization JSON with the key `visualization_json` as the key

                Important notes:

                1. **Efficient Handling of Simple Queries:**
                - For straightforward questions about room counts, lists, or connections, use the `RoomRelationshipAnalysis` tool immediately.
                - Don't overthink simple questions. Respond quickly and efficiently, avoiding unnecessary steps. 
                - If there is more than 1 item, just list it as a list in markdown format.
                - Examples of simple queries and how to handle them:
                    ```
                    Question: How many rooms are there? How many rooms are in the building? Total number of rooms
                    Thought: This is a simple room count query. I'll use RoomRelationshipAnalysis directly.
                    Action: RoomRelationshipAnalysis
                    Action Input: How many rooms are there in total?
                    Observation: [Result from the tool]
                    ```
                    ```
                    Question: Can you list all the rooms?
                    Thought: This is asking for a list of rooms. I'll use RoomRelationshipAnalysis.
                    Action: RoomRelationshipAnalysis
                    Action Input: Can you list all the rooms?
                    Observation: [Result from the tool]
                    ```

                2. **Determining the Appropriate Action:**
                - If {intent_category} is Compliance Check, use the `room-identification-chain-output` `**Generated Query:**` instead of `input`. Follow these steps:
                    a. First, use the `RoomRelationshipAnalysis` tool to check if each room exists in the data:
                        Action: RoomRelationshipAnalysis
                        Action Input: Check if these rooms exist: [List all rooms from room-identification-chain-output]
                        Observation: [Result from the tool]
                    b. Interpret the result to determine which rooms exist and which don't. Create a note about room existence:
                        Thought: I will create a note about which rooms exist and which don't.
                        Note: [List of rooms that exist] exist in the data. [List of rooms that don't exist] do not exist in the data.
                    c. If both the source and target rooms exist, use `RoomPathCalculation` to find all paths:
                        Action: RoomPathCalculation
                        Action Input: source_room: [source_room], target_room: [target_room]
                        Observation: [Result from the tool]
                    d. Use 4. **After Using RoomPathCalculation:**
                    e. Add as a note about the room and paths and make sure is part of the Final Answer

                3. **Using the RoomPathCalculation Tool:**
                - Use the exact format `source_room: <SOURCE_ROOM>, target_room: <TARGET_ROOM>` for the `Action Input`.
                - Ensure that the paths returned are relevant to the specified source and target rooms.
                - Use -> to separate each room in the path (e.g., Room1 -> Room2 -> Room3)
                - For each path, calculate and report the distance in meters (m) and ensure it is in the final answer.
                - Only consider paths through hallways and other critical areas.
                - If no relevant paths are found, inform the user.

                4. **After Using RoomPathCalculation:**
                - If there are five or fewer relevant paths, always use `RoomPathVisualization` to display the result.
                - If there are more than five paths, do not invoke `RoomPathVisualization` and inform the user that there are too many paths for visualization.
                - Review the paths returned to ensure they start from the source room and end at the target room.

                5. **Using the RoomRelationshipAnalysis Tool:**
                - Use this tool for analyzing relationships between rooms and retrieving room data, including connections.
                - You can provide queries in natural language, and the tool will interpret the intent of the query.
                - For simple queries, use the tool immediately without overthinking and return in markdown format.

                Begin!

                Question: {{input}}
                {{agent_scratchpad}}
                """

                prompt = PromptTemplate.from_template(bim_revit_agent_template)
                llm_chain = LLMChain(llm=self.bim_revit_data_model, prompt=prompt)
                agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools)

                self.room_analysis_agent = AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iterations=10
                )

            def parse_visualization_json(self, visualization_json_str):
                # Remove leading and trailing quotes, backslashes, and whitespace
                visualization_json_str = visualization_json_str.strip("'\"\n ").rstrip("}'\n ")

                # Ensure the string ends with a single closing brace
                if not visualization_json_str.endswith("}"):
                    visualization_json_str = visualization_json_str.rstrip("'}") + "}"

                try:
                    # Try to parse as JSON
                    print(f"Returning JSON after parsing second time")
                    return json.loads(visualization_json_str)
                except json.JSONDecodeError:
                    try:
                        # If JSON parsing fails, try to evaluate as a Python literal
                        print(f"Returning JSON as Python literal string after parsing three time")
                        return ast.literal_eval(visualization_json_str)
                    except (SyntaxError, ValueError):
                        print(f"Error parsing visualization JSON: {visualization_json_str[:100]}...")
                        return None

            def extract_visualization_and_answer(self, result):
                output = result.get('output', '')

                def handle_parsed_output(parsed_output):
                    final_answer = parsed_output.get('final_answer', '')
                    visualization_json = parsed_output.get('visualization_json')

                    if visualization_json not in ["None", None]:
                        # If visualization_json is a string, try to parse it
                        if isinstance(visualization_json, str):
                            visualization_json = self.parse_visualization_json(visualization_json)
                    else:
                        print(f"Visualization JSON is: {visualization_json}")
                        visualization_json = None

                    return visualization_json, final_answer

                try:
                    parsed_output = json.loads(output)
                    return handle_parsed_output(parsed_output)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, falling back to evaluate the string as a Python literal")
                    try:
                        parsed_output = ast.literal_eval(output)
                        return handle_parsed_output(parsed_output)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error evaluating as Python literal: {e}, falling back to string parsing")
                        # If JSON parsing fails, attempt to extract data using string manipulation
                        final_answer_start = output.find('"final_answer":')
                        visualization_json_start = output.find('"visualization_json":')

                        if final_answer_start != -1 and visualization_json_start != -1:
                            final_answer = output[final_answer_start:visualization_json_start].split(':', 1)[1].strip().strip('"').strip(',')
                            visualization_json = output[visualization_json_start:].split(':', 1)[1].strip()
                            visualization_json = self.parse_visualization_json(visualization_json)
                        else:
                            final_answer = output
                            visualization_json = None

                        return visualization_json, final_answer

            def process(self, query):
                self.setup_agent(query)  # Set up the agent with the current query
                intent_category = query.get('intent_category', 'Unknown')
                if intent_category == "Compliance Check":
                    modified_input = query.get('room-identification-chain-output', query.get('input'))
                    result = self.room_analysis_agent.invoke({"input": modified_input})

                    # Extract the visualization JSON and final answer from the agent's output
                    visualization_json, final_answer = self.extract_visualization_and_answer(result)

                    return {
                        "input": query.get('input'),
                        "code-regulation-rag-chain-output": query.get('code-regulation-rag-chain-output'),
                        "room-identification-chain-output": query.get('room-identification-chain-output'),
                        "bim-revit-data-chain-output": final_answer,
                        "bim-revit-data-visualisation-json-output": visualization_json
                    }
                else:
                    result = self.room_analysis_agent.invoke({"input": query})

                    # Extract the visualization JSON and final answer from the agent's output
                    visualization_json, final_answer = self.extract_visualization_and_answer(result)

                    return {
                        "input": query,
                        "bim-revit-data-chain-output": final_answer,
                        "bim-revit-data-visualisation-json-output": visualization_json
                    }

        bim_chain = BIMRevitDataChain(catalog_name, schema_name, llm_model, volume_name)
            
        def process_wrapper(query):
            result = bim_chain.process(query)
            return result
        
        return RunnableLambda(process_wrapper)

    @staticmethod
    def create_compliance_check_chain(llm_model):
        compliance_check_prompt = """
            You are an expert in building code compliance. Analyze the following information to determine if the subject complies with the relevant codes and regulations.

            **Input Query**: {input}

            **Available Data Fields**: {available_data_fields}

            **Filtered Regulatory Information**:
            {filtered_regulatory_info}

            **Room Identification**:
            {room_identification_chain_output}

            **BIM/Revit Data**:
            {bim_revit_data_chain_output}

            **BIM/Revit Visualization Data**:
            {bim_revit_vis_data}

            **Task**:
            Based on the above information, assess the compliance of the subject with the filtered regulatory information. Only assess requirements for which relevant BIM/Revit data are available. Do not assess or make assumptions about requirements for which data is lacking. Do also include the note infomrmation.

            Important: If a value is **None**, empty, or not provided, treat it as "information not available" and do not make any assumptions about it.

            When comparing rooms, ensure that room names and other relevant fields are compared in a **case-insensitive manner** (i.e., convert room names to lowercase when comparing). For each room, indicate whether it was found or not in the BIM/Revit data. Additionally, explain how the rooms are related to each other, such as their connections or proximity.

            **Assess Room Location and Entrance Together**: Room Location and Entrance should be assessed as a combined evaluation. Verify the location of the room and the proximity of its entrance to relevant areas. Ensure that the relationship between the room’s location and its entrance complies with the regulatory requirements. If data is missing for either Room Location or Entrance, mark the assessment as "Requires Further Investigation."

            **Check for Distances**: If any requirement involves distance, check the distance provided in the BIM/Revit Visualization Data (if it is **not None** or **empty**) and confirm whether it meets the requirement. If the distance data is not available, mark the requirement as "Requires Further Investigation."

            Follow these steps:

            1. **Identify the Subject**: Determine the main subject of the compliance check from the input query.

            2. **Assess Compliance**: For each requirement in the filtered regulatory information, compare the BIM/Revit data and BIM/Revit Visualization Data (if they are **not None** or **empty**) against it, stating whether it is:
            - Compliant
            - Non-Compliant
            - Requires Further Investigation (data is **None**, empty, or not available)

            3. **Provide Justification**: For each assessment, provide justification citing specific data points and explain the reasoning. If data is not available, explicitly state this.

            4. **Recommendations**: Offer actionable recommendations to address non-compliance issues or areas needing further investigation.

            **Output Format**:

            **Compliance Status**: [Compliant / Non-Compliant / Partially Compliant / Requires Further Investigation]

            **Explanation**:
            [Detailed explanation summarizing compliance status and key findings, including mention of any areas where data was not available.]

            **Assessment**:
            1. [Requirement]: [Assessment] - [Justification or "Data not available"]
            2. [Requirement]: [Assessment] - [Justification or "Data not available"]
            ...

            **Recommendations**:
            1. [Recommendation 1]  
            2. [Recommendation 2]  
            ...

            **To ensure compliance, please refer to the following documents**:
            1. [Reference 1 and relevant URL from the {filtered_regulatory_info}]
            2. [Reference 2 and relevant URL from the {filtered_regulatory_info}]
            ...

            **Note**: Interpret the regulatory information correctly and base your assessment solely on the filtered regulatory information and the provided BIM/Revit data and BIM/Revit Visualization Data (if they are **not None** or **empty**). If data is missing (None, empty, or not provided), explicitly state this in your assessment and classify it as "Requires Further Investigation."
        """

        compliance_check_prompt_template = ChatPromptTemplate.from_template(compliance_check_prompt)
        compliance_check_chain = LLMChain(llm=llm_model, prompt=compliance_check_prompt_template)

        def format_visualization_data(vis_data: Dict[str, Any]) -> str:
            formatted_vis_data = "**Nodes (Rooms):**\n"
            for node in vis_data['nodes']:
                formatted_vis_data += (f"- ID: {node['id']}, Name: {node['name']}, "
                                    f"Level: {node['level']}, Area: {node['area']}\n")

            formatted_vis_data += "\n**Links (Connections):**\n"
            for link in vis_data['links']:
                formatted_vis_data += (f"- From: {link['source_name']} To: {link['target_name']}, "
                                    f"Distance: {link['route_distance']}m\n"
                                    f"  Path: {' → '.join(link['path'])}\n")

            return formatted_vis_data

        def filter_regulatory_requirements(regulatory_info: str, available_data_fields: List[str]) -> str:
            requirements = regulatory_info.split('\n\n')
            filtered_requirements = [req for req in requirements if any(field.lower() in req.lower() for field in available_data_fields)]
            return '\n\n'.join(filtered_requirements)

        def process_chain_output(inputs: Dict[str, Any]) -> Dict[str, Any]:
            # Format visualization data
            if inputs['bim_revit_data_visualisation_json_output'] not in ["None", None]:
                inputs['bim_revit_data_visualisation_json_output'] = json.dumps(inputs['bim_revit_data_visualisation_json_output'])
                bim_revit_vis_data = json.loads(inputs['bim_revit_data_visualisation_json_output'])
                inputs['bim_revit_vis_data'] = format_visualization_data(bim_revit_vis_data)
            else:
                inputs['bim_revit_vis_data'] = None

            # Define available data fields based on your data
            available_data_fields = ['Distance', 'Path', 'Size', 'Area', 'Level', 'Entrance', 'please refer to the following documents:']

            # Filter the regulatory requirements
            filtered_regulatory_info = filter_regulatory_requirements(inputs['code_regulation_rag_chain_output'], available_data_fields)
            inputs['filtered_regulatory_info'] = filtered_regulatory_info

            # Add available data fields to inputs for the prompt
            inputs['available_data_fields'] = ', '.join(available_data_fields)

            # Prepare the inputs for the prompt
            prompt_inputs = {
                'input': inputs['input'],
                'available_data_fields': inputs['available_data_fields'],
                'filtered_regulatory_info': inputs['filtered_regulatory_info'],
                'room_identification_chain_output': inputs['room_identification_chain_output'],
                'bim_revit_data_chain_output': inputs['bim_revit_data_chain_output'],
                'bim_revit_vis_data': inputs['bim_revit_vis_data'],
            }

            # Run the chain
            chain_output = compliance_check_chain.run(prompt_inputs)

            return {
                'input': inputs['input'],
                'code_regulation_rag_chain_output': inputs['code_regulation_rag_chain_output'],
                'filtered_regulatory_info': filtered_regulatory_info,
                'room_identification_chain_output': inputs['room_identification_chain_output'],
                'bim_revit_data_chain_output': inputs['bim_revit_data_chain_output'],
                'bim_revit_data_visualisation_json_output': inputs['bim_revit_data_visualisation_json_output'],
                'compliance_check_chain_output': chain_output
            }

        return RunnableLambda(process_chain_output)

    @staticmethod
    def create_general_response_chain(llm_model):
        general_prompt = PromptTemplate.from_template("""
        You are ReguBIM AI, a GenAI-powered compliance assistant that simplifies the building compliance process. You integrate graph analysis of BIM data with building codes and regulations, enabling users to query both regulatory requirements and BIM data effortlessly, streamlining the compliance process for BIM models.

        You are an assistant for a building information system. The user has asked a question that doesn't fit into our specific categories of building codes, BIM data, or compliance checks. Please provide a helpful and friendly response, and if possible, guide them towards asking a question that our system can handle.
        User query: {query}
        Response:
        """)
        
        chain = LLMChain(llm=llm_model, prompt=general_prompt)
        
        def process_query(input_dict):
            query = input_dict.get('query', '')
            result = chain.invoke({"query": query})
            return {
                "input": query,
                "output": result["text"]
            }
        
        return RunnableLambda(process_query)
    
    def debug_wrapper(self, runnable, step_name):
        def wrapper(x):
            self.debug_print(f"Debug - {step_name} input: {x}\n")
            try:
                result = runnable(x)
                self.debug_print(f"Debug - {step_name} output: {result}\n")
                return result
            except Exception as e:
                error_msg = f"Error in {step_name}: {str(e)}\n"
                self.debug_print(error_msg)
                return {"error": error_msg}
        return wrapper
    
    def load_context(self):
        
        # Reconstruct the llm_model
        llm_model = ChatDatabricks(endpoint=self.llm_model_name, max_tokens=3000, temperature=0.0)

        # Create individual chains and assign to self
        self.chain_query_classification = self.create_chain_query_classification(llm_model)
        self.code_regulation_rag_chain = self.create_code_regulation_rag_chain(llm_model)
        self.room_identification_chain = self.create_room_identification_chain(llm_model)
        self.bim_revit_data_chain = self.create_bim_revit_data_chain(self.catalog_name, self.schema_name, llm_model, self.volume_name)
        self.compliance_check_chain = self.create_compliance_check_chain(llm_model)
        self.chain_general_response = self.create_general_response_chain(llm_model)

    def multi_stage_chain(self, x):
        # Check if x is a list and extract the first item
        if isinstance(x, list):
            x = x[0]

        query = x["query"]

        intent_result = self.debug_wrapper(
            self.chain_query_classification.invoke,
            "chain_query_classification"
        )({"query": query})

        intent_category = intent_result.get('output')

        if intent_category == "Building Codes and Regulations":
            return self.debug_wrapper(
                self.code_regulation_rag_chain.invoke,
                "code_regulation_rag_chain"
            )(query)

        elif intent_category == "BIM Revit Data":
            bim_revit_input = {
                'input': query,
                'intent_category': intent_category
            }
            return self.debug_wrapper(
                self.bim_revit_data_chain.invoke,
                "bim_revit_data_chain"
            )(bim_revit_input)

        elif intent_category == "Compliance Check":
            return self.compliance_check_full_chain(x, intent_category)

        else:
            return self.debug_wrapper(
                self.chain_general_response.invoke,
                "chain_general_response"
            )({"query": query})

    def compliance_check_full_chain(self, x, intent_category):
        initial_input = x

        code_reg_result = self.debug_wrapper(
            lambda c : self.code_regulation_rag_chain.invoke(c['query']),
            "code_regulation_rag_chain"
        )(initial_input)

        room_identification_input = {
            'input': initial_input,
            'code-regulation-rag-chain-output': code_reg_result
        }
        room_identification_result = self.debug_wrapper(
            lambda r : self.room_identification_chain(
                {'input': r['code-regulation-rag-chain-output']['input'], 
                 'code-regulation-rag-chain-output': r['code-regulation-rag-chain-output']['output']}),
            "room_identification_chain"
        )(room_identification_input)

        bim_revit_input = {
            'input': room_identification_result['input'],
            'intent_category': intent_category,
            'code-regulation-rag-chain-output': room_identification_result['code-regulation-rag-chain-output'],
            'room-identification-chain-output': room_identification_result['room-identification-chain-output']
        }
        bim_revit_result = self.debug_wrapper(
            self.bim_revit_data_chain.invoke,
            "bim_revit_data_chain"
        )(bim_revit_input)

        compliance_check_input = {
            'input': bim_revit_result['input'],
            'code_regulation_rag_chain_output': bim_revit_result['code-regulation-rag-chain-output'],
            'room_identification_chain_output': bim_revit_result['room-identification-chain-output'],
            'bim_revit_data_chain_output': bim_revit_result['bim-revit-data-chain-output'],
            'bim_revit_data_visualisation_json_output': bim_revit_result['bim-revit-data-visualisation-json-output']
        }
        compliance_check_result = self.debug_wrapper(
            self.compliance_check_chain.invoke,
            "compliance_check_chain"
        )(compliance_check_input)

        return compliance_check_result

    def predict(self, model_input: pd.DataFrame) -> List[str]:

        if not hasattr(self, 'chain_query_classification'):
            self.load_context()

        # Check if 'debug_mode' column exists and set DEBUG_MODE accordingly
        if 'debug_mode' in model_input.columns:
            # Use the first value in 'debug_mode' column to set DEBUG_MODE
            self.DEBUG_MODE = bool(model_input['debug_mode'].iloc[0])
        else:
            # Default value if not provided
            self.DEBUG_MODE = False

        results = []
        for _, input_row in model_input.iterrows():
            input_item = input_row.to_dict()
            result = self.multi_stage_chain(input_item)
            if self.DEBUG_MODE:
                print(f"Debug - Final output: {result}\n")
            result_json = json.dumps(result)
            results.append(result_json)
        return results

# COMMAND ----------

llm_model_name = "databricks-meta-llama-3-1-70b-instruct"
multi_intent_regulation_bim_compliance_wrapper = MultiStageSystemWrapper(llm_model_name, catalog_name, schema_name,volume_name)

# COMMAND ----------

import json
import pandas as pd

# Define widget to accept the input parameter 'query'

dbutils.widgets.text("query_text", "Hello how are you?")

# Retrieve the input parameter from widgets
input_query = dbutils.widgets.get("query_text")

# Create the input data for the model
query_df = pd.DataFrame([{"query": input_query, "debug_mode": False}])

# Get prediction (replace this with your actual model call)
result = multi_intent_regulation_bim_compliance_wrapper.predict(query_df)

# Convert the result to a JSON-compatible format (if needed, depends on result structure)
result_dict = json.loads(result[0])

# Convert the result to JSON string
result_json = json.dumps(result_dict, indent=4)

# Use dbutils to exit and return the result as the job output
dbutils.notebook.exit(result_json)
