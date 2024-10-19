# Standard library imports
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


class RoomPathCalculationTool:
    def __init__(self, data_retrieval_tool, catalog_name, schema_name):
        self.data_retrieval_tool = data_retrieval_tool
        

    def calculate(self, input_str):
        try:
            # Split the input into source_room and target_room
            source_room, target_room = [item.split(":")[1].strip() for item in input_str.split(",")]
            
            room_edges_df = self.data_retrieval_tool.get_room_edges()
            result = self.find_all_shortest_paths(room_edges_df, source_room.upper(), target_room.upper())
            
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

    def generate_visualization_json(self, room_graph_data_json):
        room_graph_data_json = room_graph_data_json.replace("\'\n","").replace("\'{\"nodes\":","{\"nodes\":")
        
        return room_graph_data_json.strip("'")

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
            x_sum = sum(float(p['X']) for p in points)
            y_sum = sum(float(p['Y']) for p in points)
            z_sum = sum(float(p['Z']) for p in points)
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

    def find_all_shortest_paths(self, room_edges_df, source_room, target_room):
        
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