# Databricks notebook source
# MAGIC %md
# MAGIC #### Dependency and Installation

# COMMAND ----------

# MAGIC %run ../common/installation_setup

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tools and ReAct Agent Revit BIM Chain

# COMMAND ----------

# Standard library imports
import os
import re
import ast
import json
import logging
from typing import List, Dict, Any, Union
import logging
logging.getLogger("py4j").setLevel(logging.WARNING)

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

# MLflow imports
import mlflow
import mlflow.pyfunc
import cloudpickle
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

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
        - If there are five or fewer relevant paths, use `RoomPathVisualization` to display the result.
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
            verbose=True,
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


# COMMAND ----------

def create_bim_revit_data_chain(catalog_name, schema_name, llm_model, volume_name):
    
    bim_chain = BIMRevitDataChain(catalog_name, schema_name, llm_model, volume_name)
        
    def process_wrapper(query):
        result = bim_chain.process(query)
        return result
    
    return RunnableLambda(process_wrapper)


# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

llm_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=3000, temperature=0.0)
bim_revit_data_chain = create_bim_revit_data_chain(catalog_name, schema_name, llm_model, volume_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test the Agent

# COMMAND ----------

# Example question to pass to your process_question function
bim_revit_question_1 = {"query": "What is the total number of rooms and the list of room names in the building?", "intent_category": "BIM Revit Data"}

# Call the function to get the answer
bim_revit_answer_1 = bim_revit_data_chain.invoke(bim_revit_question_1)

# COMMAND ----------

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

print_nested_dict_display(bim_revit_answer_1)

# COMMAND ----------

# Example question to pass to your process_question function
bim_revit_question_2 = {"query": "What is the total number of paths in the building?", "intent_category": "BIM Revit Data"}

# Call the function to get the answer
bim_revit_answer_2 = bim_revit_data_chain.invoke(bim_revit_question_2)

# COMMAND ----------

print_nested_dict_display(bim_revit_answer_2)

# COMMAND ----------

# Example question to pass to your process_question function
bim_revit_question_3 = {"query": "What are the paths from Fcc to staircase?", "intent_category": "BIM Revit Data"}

# Call the function to get the answer
bim_revit_answer_3 = bim_revit_data_chain.invoke(bim_revit_question_3)

# COMMAND ----------

print_nested_dict_display(bim_revit_answer_3)

# COMMAND ----------

html_room_path_graph_template = """
<!doctype html>
<html lang="en">
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0" />
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
        <link
            href="https://fonts.googleapis.com/css?family=Roboto:400,700&subset=latin,cyrillic-ext"
            rel="stylesheet"
            type="text/css" />
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css" rel="stylesheet" />
        <script src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
        <script src="https://d3js.org/d3.v6.min.js"></script>
        <script src="https://unpkg.com/d3-v6-tip@1.0.6/build/d3-v6-tip.js"></script>
        <title>Room Route Graph Visualization</title>
        <style>
            .graph-container {
                margin: auto;
                width: 90%;
                padding: 10px;
            }
            div#data_vis_display {
                overflow: auto;
            }
            .d3-tip {
                line-height: 1.4;
                padding: 12px;
                background: rgba(0, 0, 0, 0.8);
                color: #fff;
                border-radius: 2px;
                pointer-events: none !important;
            }
            svg {
                border: 1px solid black;
            }
            .legend {
                font-size: 12px;
                font-family: sans-serif;
            }
            .route-label {
                font-size: 10px;
                font-weight: bold;
                fill: #fff;
                text-anchor: middle;
                dominant-baseline: central;
            }
        </style>
    </head>
    <body>
        <div class="graph-container">
            <div id="data_vis_display"></div>
        </div>

        <script>
            $(document).ready(function () {
                var graph = """ + json.dumps(bim_revit_answer_3['bim-revit-data-visualisation-json-output']) + """;

                const width = 1000,
                    height = 600,
                    circle_radius = 40;
                const svg = d3.select("#data_vis_display").append("svg").attr("viewBox", `0 0 ${width} ${height}`);

                const nodeColorScale = d3
                    .scaleOrdinal()
                    .domain(["source", "destination", "intermediate"])
                    .range(["#1f77b4", "#2ca02c", "#ff7f0e"]);

                const tip = d3
                    .tip()
                    .attr("class", "d3-tip")
                    .offset([-10, 0])
                    .html(function (event, d) {
                      
                        if (d.route) {
                          console.log(d)
                            return `<strong>Route ${d.route}</strong><br>
                                Path: ${d.path.join(" → ")}<br>
                                Distance: ${d.route_distance.toFixed(2)}m`;
                        } else {
                            return `<strong>${d.name}</strong><br>
                                Type: ${d.type}<br>
                                Area: ${d.area}<br>
                                Level: ${d.level}`;
                        }
                    });

                svg.call(tip);

                // Arrow marker definition
                svg.append("defs")
                    .append("marker")
                    .attr("id", "end")
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 10)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .style("fill", "#666")
                    .style("stroke", "none");

                const simulation = d3
                    .forceSimulation()
                    .force(
                        "link",
                        d3
                            .forceLink()
                            .id((d) => d.id)
                            .distance(200)
                    )
                    .force("charge", d3.forceManyBody().strength(-500))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collide", d3.forceCollide().radius(circle_radius * 1.5));

                const linkGroup = svg.append("g").attr("class", "links");
                const nodeGroup = svg.append("g").attr("class", "nodes");

                function getNodeTypeFromLinks(nodeId, links) {
                    let sourceCount = 0;
                    let targetCount = 0;

                    // Iterate over all links to count occurrences of nodeId as source and target
                    links.forEach((link) => {
                        if (link.source === nodeId) {
                            sourceCount++;
                        }
                        if (link.target === nodeId) {
                            targetCount++;
                        }
                    });

                    console.log(nodeId);
                    // Determine the type based on the counts
                    if (targetCount > 1) {
                        return "intermediate";
                    } else if (sourceCount > 0) {
                        return "source";
                    } else if (targetCount > 0) {
                        return "destination";
                    }

                    return "unknown"; // Fallback if the nodeId is not found in the links
                }

                // Function to assign unique link numbers to overlapping links
                function computeLinkNumbers(links) {
                    let linkGroups = {};
                    links.forEach(function (d) {
                        let sourceId = typeof d.source === "object" ? d.source.id : d.source;
                        let targetId = typeof d.target === "object" ? d.target.id : d.target;
                        let key = [sourceId, targetId].sort().join(",");
                        if (!linkGroups[key]) {
                            linkGroups[key] = [];
                        }
                        linkGroups[key].push(d);
                    });
                    for (let key in linkGroups) {
                        let group = linkGroups[key];
                        group.forEach(function (link, i) {
                            link.linknum = i;
                            link.totalLinks = group.length;
                        });
                    }
                }

                // Call the function to assign link numbers
                computeLinkNumbers(graph.links);

                function update() {
                    const link = linkGroup.selectAll("g").data(graph.links).join("g");

                    // Append the path first
                    link.append("path")
                        .attr("fill", "none")
                        .attr("stroke", "#999")
                        .attr("stroke-width", 2)
                        .attr("marker-end", "url(#end)");

                    // Append labels
                    const labelGroup = link.append("g").attr("class", "label");

                    labelGroup
                        .append("rect")
                        .attr("width", 50)
                        .attr("height", 30)
                        .attr("rx", 10)
                        .attr("ry", 10)
                        .attr("fill", "#fff")
                        .attr("stroke", "#999");

                    labelGroup
                        .append("text")
                        .attr("class", "route-label")
                        .attr("dy", ".35em")
                        .style("font-size", "12px")
                        .style("font-weight", "bold")
                        .style("fill", "#000")
                        .attr("text-anchor", "middle")
                        .text((d) => `Route ${d.route}`)
                      .on("mouseover", tip.show)
                        .on("mouseout", tip.hide);

                    const node = nodeGroup
                        .selectAll("g")
                        .data(graph.nodes)
                        .join("g")
                        .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended))
                        .on("mouseover", tip.show)
                        .on("mouseout", tip.hide);

                    node.append("circle")
                        .attr("r", circle_radius)
                        .attr("fill", (d) => nodeColorScale(getNodeTypeFromLinks(d.id, graph.links)));

                    node.append("text")
                        .attr("text-anchor", "middle")
                        .attr("dominant-baseline", "central")
                        .attr("font-family", "FontAwesome")
                        .style("fill", "white")
                        .each(function (d) {
                            const textElement = d3.select(this);
                            const icon = getIconFromName(d.name);
                            textElement.append("tspan").attr("font-size", "20px").attr("dy", "-0.5em").text(icon);

                            const words = d.name.split(/\s+/);
                            let lineHeight = 1.1;
                            words.forEach((word, i) => {
                                textElement
                                    .append("tspan")
                                    .attr("x", 0)
                                    .attr("dy", i ? `${lineHeight}em` : "1.5em")
                                    .attr("font-size", "10px")
                                    .text(word);
                            });
                        });

                    simulation.nodes(graph.nodes).on("tick", ticked);
                    simulation.force("link").links(graph.links);
                    simulation.alpha(1).restart();

                    // Legend
                    const legend = svg.append("g").attr("class", "legend").attr("transform", "translate(20,20)");

                    const legendData = [
                        { type: "Source", color: nodeColorScale("source") },
                        { type: "Destination", color: nodeColorScale("destination") },
                        { type: "Intermediate", color: nodeColorScale("intermediate") }
                    ];

                    const legendItems = legend
                        .selectAll(".legend-item")
                        .data(legendData)
                        .enter()
                        .append("g")
                        .attr("class", "legend-item")
                        .attr("transform", (d, i) => `translate(0,${i * 20})`);

                    legendItems
                        .append("rect")
                        .attr("width", 18)
                        .attr("height", 18)
                        .style("fill", (d) => d.color);

                    legendItems
                        .append("text")
                        .attr("x", 24)
                        .attr("y", 9)
                        .attr("dy", ".35em")
                        .text((d) => d.type)
                        .style("font-size", "12px");
                }
              
               function manhattanDistance(point1, point2) {
                    return Math.abs(point1.x - point2.x) + Math.abs(point1.y - point2.y);
                }

                function linkPath(d) {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const dr = Math.sqrt(dx * dx + dy * dy);

                    const angle = Math.atan2(dy, dx);

                    // Calculate offset angle for this link
                    const totalLinks = d.totalLinks;
                    const linknum = d.linknum;

                    const angleOffset = (linknum - (totalLinks - 1) / 2) * (Math.PI / 12);

                    // Calculate start and end points on the circles' circumferences
                    const sourceAngle = angle + angleOffset;
                    const targetAngle = angle + Math.PI + angleOffset;

                    const sourceX = d.source.x + Math.cos(sourceAngle) * circle_radius;
                    const sourceY = d.source.y + Math.sin(sourceAngle) * circle_radius;
                    const targetX = d.target.x + Math.cos(targetAngle) * circle_radius;
                    const targetY = d.target.y + Math.sin(targetAngle) * circle_radius;

                    // Define curvature
                    const curvature = 0.25 * (linknum - (totalLinks - 1) / 2);

                    // Calculate the path
                    const path = `M${sourceX},${sourceY}A${dr * Math.abs(curvature)},${dr * Math.abs(curvature)} 0 0,${
                        curvature > 0 ? 1 : 0
                    } ${targetX},${targetY}`;

                    return path;
                }

                function ticked() {
                    linkGroup.selectAll("g").each(function (d) {
                        const link = d3.select(this);
                        const path = linkPath(d);

                        const pathElement = link.select("path").attr("d", path).node();

                        // Ensure path element exists before calculating its length
                        if (pathElement) {
                            const totalLength = pathElement.getTotalLength();
                            const midPoint = pathElement.getPointAtLength(totalLength / 2);

                            // Update label positions
                            link.select(".label").attr("transform", `translate(${midPoint.x},${midPoint.y})`);

                            link.select(".label rect").attr("x", -25).attr("y", -15);

                            link.select(".label text").attr("x", 0).attr("y", 0);
                        }
                    });

                    nodeGroup.selectAll("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
                }

                function dragstarted(event, d) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }

                function dragged(event, d) {
                    d.fx = event.x;
                    d.fy = event.y;
                }

                function dragended(event, d) {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }

                function getIconFromName(name) {
                    const iconMap = {
                        Staircase: "\ue289",
                        Computer: "\ue4e5",
                        Electrical: "\uf0eb",
                        Washroom: "\uf7d8",
                        Meeting: "\ue537",
                        Training: "\uf51c",
                        Hallway: "\uf557",
                        Smoking: "\uf48d",
                        Security: "\ue54a",
                        Prayer: "\uf683",
                        Mechanical: "\uf0ad",
                        Cafeteria: "\uf0f4",
                        Outside: "\uf850",
                        Loading: "\uf4de"
                    };
                    const trimmedName = name.trim().toLowerCase();
                    const matchingKeyword = Object.keys(iconMap).find((keyword) =>
                        trimmedName.includes(keyword.toLowerCase())
                    );
                    return iconMap[matchingKeyword] || "\uf0db";
                }

                update();
            });
        </script>
    </body>
</html>

"""

# COMMAND ----------

# Render the HTML content directly inside Databricks
displayHTML(html_room_path_graph_template)
