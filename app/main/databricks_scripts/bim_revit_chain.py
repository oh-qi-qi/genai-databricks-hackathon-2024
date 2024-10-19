# Standard library imports
import os
import re
import ast
import json
import sys
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

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
sys.path.insert(0, main_dir)

from bim_revit_agent.bim_revit_agent_setup import BIMRevitAgent
    

from common.utils import print_nested_dict_display

# Usage example
from common.databricks_config import (
    DATABRICKS_URL, 
    TOKEN, 
    DATABRICKS_WAREHOUSE_ID,
    catalog_name, 
    schema_name
)


def create_bim_revit_data_chain(catalog_name, schema_name,llm_model, databricks_url, token, databricks_warehouse_id):
    
    bim_revit_agent = BIMRevitAgent(catalog_name, schema_name,llm_model, databricks_url, token, databricks_warehouse_id)
        
    def process_wrapper(query):
        result = bim_revit_agent.process(query)
        return result
    
    return RunnableLambda(process_wrapper)

llm_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=3000, temperature=0.0)
bim_revit_data_chain = create_bim_revit_data_chain(catalog_name, schema_name, llm_model, DATABRICKS_URL, TOKEN, DATABRICKS_WAREHOUSE_ID)

bim_revit_question_1 = {"query": "What is the total number of rooms and the list of room names in the building?", "intent_category": "BIM Revit Data"}

# Call the function to get the answer
bim_revit_answer_1 = bim_revit_data_chain.invoke(bim_revit_question_1)
print_nested_dict_display(bim_revit_answer_1)

#bim_revit_question_2 = {"query": "What is the total number of paths?", "intent_category": "BIM Revit Data"}
#
## Call the function to get the answer
#bim_revit_answer_2 = bim_revit_data_chain.invoke(bim_revit_question_2)
#print_nested_dict_display(bim_revit_answer_2)

#bim_revit_question_3 = {"query": "What are the paths from Fcc to staircase?", "intent_category": "BIM Revit Data"}
#
## Call the function to get the answer
#bim_revit_answer_3 = bim_revit_data_chain.invoke(bim_revit_question_3)
#print_nested_dict_display(bim_revit_answer_3)