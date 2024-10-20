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


intents = {
            "count_rooms": ["how many rooms", "how many rooms in the building", "count rooms", "total rooms"],
            "count_connections": ["how many connections", "count connections", "total connections", "total number of connections", "number of connections", "total number of paths", "number of paths", "how many paths", "total paths"],
            "list_rooms": ["list rooms", "show rooms", "what rooms", "room list", "list rooms in the building"],
            "check_rooms_exist": ["check rooms exist", "do these rooms exist", "are these rooms present", "check if the following rooms exist", "check if the rooms exist", "check if these rooms exist"]
        }

def interpret_query(query):
    query = query.lower()
    all_phrases = [phrase for phrases in intents.values() for phrase in phrases]
    best_match = process.extractOne(query, all_phrases)

    print(best_match)
    if best_match[1] >= 70:
        matched_phrase = best_match[0]
        for intent, phrases in intents.items():
            if matched_phrase in phrases:
                return intent
    
    
#intent = interpret_query("What is the total number of paths in the building?")
#print(intent)

room_names_lower = ['fcc', 'staircase']
existing_rooms_lower = ['mechanical 01-22', 'female washroom 01-9', 'internal staircase 01-30', 'staircase 03-12', 'security room with raised floor 01-13', 'electrical 02-2', 'prayer room 01-18', 'training room 01-31', 'mdf 01-20', 'male washroom 03-9', 'staircase 03-3', 'mechanical 03-1', 'pwd washroom 03-4', '6pax meeting room 01-28', 'spare room 01-25', 'hallway 01-33', 'electrical 01-26', 'female washroom 01-15', 'smoking area 01-5', 'female washroom 03-10', 'pwd washroom 02-4', '4pax meeting room 01-11', 'male washroom 01-14', 'female washroom 02-10', 'electrical 03-13', 'male washroom 02-9', 'internal staircase 03-8', 'female washroom 03-5', 'staircase 01-17', 'electrical 03-2', 'pwd washroom 03-11', 'pwd washroom 02-11', 'electrical 02-13', 'male washroom 01-10', 'male washroom 02-6', 'internal staircase 02-8', 'electrical 01-23', 'hallway 01-34', 'staircase 02-3', 'female washroom 02-5', 'male washroom 03-6', 'pwd washroom 01-8', 'mechanical 02-1', 'pwd washroom 01-16', 'cafeteria 01-7', 'fcc 01-21', 'protected staircase 01-24', 'staircase 02-12', 'training room 01-32', 'computer room 01-19', 'outside', 'office area 03-15', 'office area 02-14', 'loading area 01-6']

existing = []
non_existing = []

# Loop through each room name in the lowercased room list
for room_lower in room_names_lower:
    # Extract all matches with scores (no cutoff here, we'll filter manually)
    matches = process.extract(room_lower, existing_rooms_lower)
    
    # Check each match
    for match_name, match_score in matches:
        if match_score > 80:  # Only add matches with a score greater than 80
            # Find the original (non-lowercased) room name in the list and append to existing
            existing.append(match_name)
        else:
            # Add to non-existing if no match meets the threshold
            non_existing.append(room_lower)

print("Existing:", existing)
print("Non-existing:", non_existing)

## Usage example
#from common.databricks_config import (
#    DATABRICKS_URL, 
#    TOKEN, 
#    DATABRICKS_WAREHOUSE_ID,
#    catalog_name, 
#    schema_name
#)
#
#llm_model_name = "databricks-meta-llama-3-1-70b-instruct"
#multi_chain_wrapper = MultiStageSystemWrapper(llm_model_name, catalog_name, schema_name, DATABRICKS_URL, TOKEN, DATABRICKS_WAREHOUSE_ID)

#query_1 = pd.DataFrame([
#    {"query": "What are FCC Room Requirements I have to comply with?", "debug_mode": True}
#])
#result_1 = multi_chain_wrapper.predict(model_input=query_1)
#print_nested_dict_display(json.loads(result_1[0]))

#query_2 = pd.DataFrame([
#    {"query": "What is the path from FCC to Staircase?", "debug_mode": True}
#])
#result_2 = multi_chain_wrapper.predict(query_2)
#print_nested_dict_display(json.loads(result_2[0]))

#query_3 = pd.DataFrame([
#    {"query": "Does the FCC comply with code and regulation?", "debug_mode": True}
#])
#result_3 = multi_chain_wrapper.predict(query_3)
#print_nested_dict_display(json.loads(result_3[0]))
#
#query_4 = pd.DataFrame([
#    {"query": "Hello, can you help me?", "debug_mode": False}
#])
#result_4 = multi_chain_wrapper.predict(query_4)
#print_nested_dict_display(json.loads(result_4[0]))