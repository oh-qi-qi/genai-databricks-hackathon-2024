# Standard library imports
import re
import ast
import json
import logging
from typing import List, Dict, Any, Union

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx
import inflection
from fuzzywuzzy import process  # FuzzyWuzzy (for string matching)

# LangChain imports
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_retrieval_tool import DataRetrievalTool
from room_path_calculation_tool import RoomPathCalculationTool
from room_relationship_tool import RoomRelationshipTool
from table_generation_tool import TableGenerationTool


class BIMRevitAgent:
    def __init__(self, catalog_name, schema_name,llm_model, databricks_url, token, databricks_warehouse_id):
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.bim_revit_data_model = llm_model
        self.databricks_url = databricks_url
        self.token = token
        self.databricks_warehouse_id = databricks_warehouse_id
        self.setup_tools()
        self.agent = None  # We'll set this up later

    def setup_tools(self):
        self.data_retrieval_tool = DataRetrievalTool(self.databricks_url, self.token, self.databricks_warehouse_id, self.catalog_name, self.schema_name)
        self.path_calculation_tool = RoomPathCalculationTool(self.data_retrieval_tool, self.catalog_name, self.schema_name)
        self.room_relationship_tool = RoomRelationshipTool(self.data_retrieval_tool, self.catalog_name, self.schema_name)
        self.table_generation_tool = TableGenerationTool()

        self.tools = [
            Tool(
                name="RoomPathCalculation",
                func=self.path_calculation_tool.calculate,
                description="Calculates the path between two rooms. Input must be in the format 'source_room: <SOURCE_ROOM>, target_room: <TARGET_ROOM>'. Room names should be in uppercase."
            ),
            Tool(
                name="RoomPathVisualization",
                func=self.path_calculation_tool.generate_visualization_json,
                description="Generates and returns a visualization json of the path between rooms. Input should be the graph JSON returned by RoomPathCalculation. Returns the visualization JSON."
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
        Visualization JSON: only the valid JSON format data for visualization, or 'None' if not applicable
        Final Answer:  the final answer to the question and include any relevant note as the value and the key is `final_answer` and the Visualization JSON with the key `visualization_json` as the key
        
        Important notes:

        1. **Efficient Handling of Simple Queries:**
        - For straightforward questions about room counts, lists, connections or paths, use the `RoomRelationshipAnalysis` tool immediately.
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
        - If {intent_category} is BIM Revit Data and is related to finding paths between rooms, use the `RoomPathCalculation` tool
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
        visualization_json_str = visualization_json_str.strip("'\"\n ").rstrip("}'\"\n ")
        
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
            final_answer = parsed_output.get('final_answer', '').rstrip("}'\"\n ")
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

                def find_key(output, key):
                    double_quoted = output.find(f'"{key}":')
                    single_quoted = output.find(f"'{key}':")
                    return max(double_quoted, single_quoted)

                final_answer_start = find_key(output, "final_answer")
                visualization_json_start = find_key(output, "visualization_json")

                if final_answer_start != -1 and visualization_json_start != -1:
                    # Extract final_answer
                    final_answer_end = visualization_json_start
                    final_answer = output[final_answer_start:final_answer_end].split(':', 1)[1].strip()
                    final_answer = final_answer.strip('"').strip("'").strip(',').strip()

                    # Extract visualization_json
                    visualization_json = output[visualization_json_start:].split(':', 1)[1].strip()
                    # Remove trailing characters that might have been included
                    visualization_json = visualization_json.rstrip("}'\"")
                    visualization_json = self.parse_visualization_json(visualization_json)
                else:
                    # If we can't find both keys, treat the whole output as final_answer
                    final_answer = output.strip()
                    visualization_json = None

                return visualization_json, final_answer.rstrip("}'\"\n ")
                
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
