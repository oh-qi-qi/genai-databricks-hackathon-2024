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
sys.path.insert(0, current_dir)

from bim_revit_agent.bim_revit_agent_setup import BIMRevitAgent
    

def create_bim_revit_data_chain(catalog_name, schema_name,llm_model, databricks_url, token, databricks_warehouse_id):
    
    bim_revit_agent = BIMRevitAgent(catalog_name, schema_name,llm_model, databricks_url, token, databricks_warehouse_id)
        
    def process_wrapper(query):
        result = bim_revit_agent.process(query)
        return result
    
    return RunnableLambda(process_wrapper)