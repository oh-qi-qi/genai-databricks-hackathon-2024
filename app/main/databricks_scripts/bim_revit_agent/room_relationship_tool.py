# Stadard library imports
import os
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

class RoomRelationshipTool:
    def __init__(self, data_retrieval_tool, catalog_name, schema_name):
        self.data_retrieval_tool = data_retrieval_tool
        
        # Define intents and their associated phrases
        self.intents = {
            "count_rooms": ["how many rooms", "how many rooms in the building", "count rooms", "total rooms"],
            "count_connections": ["how many connections", "count connections", "total connections", "total number of connections", "number of connections", "total number of paths", "number of paths", "how many paths", "total paths"],
            "list_rooms": ["list rooms", "show rooms", "what rooms", "room list", "list rooms in the building"],
            "check_rooms_exist": ["check rooms exist", "do these rooms exist", "are these rooms present", "check if the following rooms exist", "check if the rooms exist", "check if these rooms exist"]
        }

    def analyze(self, query=None):
        vertices = self.data_retrieval_tool.get_room_vertices()
        edges = self.data_retrieval_tool.get_room_edges()
        
        room_graph_data_json = self.create_graph_json(vertices, edges)
        room_graph = self.create_nx_graph(vertices, edges)
        analysis_result = self.perform_analysis(room_graph, query)
        
        return room_graph, room_graph_data_json, analysis_result

    def create_nx_graph(self, vertices, edges):
        G = nx.MultiGraph()

        # Add nodes
        for _, row in vertices.iterrows():
            name = str(row['name'])
            G.add_node(name, **{k: str(v) for k, v in row.items()})

        # Create a set of valid room names
        valid_rooms = set(G.nodes())

        # Add edges based on doors
        door_count = 0
        for _, row in edges.iterrows():
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
            term_used = "connections"
            if "path" in query:
                term_used = "paths"
            return f"Total number of {term_used}: {room_graph.number_of_edges()}"
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
            for row in vertices.itertuples()
#            for row in vertices.toPandas().itertuples()
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
            for row in edges.itertuples()
#            for row in edges.toPandas().itertuples()
        ]

        room_graph_data = {
            "nodes": nodes,
            "links": links
        }

        # Minified JSON
        return json.dumps(room_graph_data, ensure_ascii=False, separators=(',', ':'))
