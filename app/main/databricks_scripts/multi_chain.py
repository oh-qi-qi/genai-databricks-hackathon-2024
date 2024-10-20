import os
import re
import ast
import json
import logging
import time
import concurrent.futures
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd

from langchain_community.chat_models import ChatDatabricks
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.schema.runnable import RunnableMap, RunnableBranch, RunnablePassthrough
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common.utils import print_nested_dict_display

from databricks_scripts.intent_chain import create_chain_query_classification
from databricks_scripts.code_regulations_chain import create_code_regulation_rag_chain
from databricks_scripts.room_identification_chain import create_room_identification_chain
from databricks_scripts.bim_revit_chain import create_bim_revit_data_chain
from databricks_scripts.compliance_chain import create_compliance_check_chain
from databricks_scripts.general_response_chain import create_general_response_chain

class MultiStageSystemWrapper:
    def __init__(self, llm_model_name, catalog_name, schema_name, databricks_url, token, databricks_warehouse_id):
        self.llm_model_name = llm_model_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.databricks_url = databricks_url
        self.token = token
        self.databricks_warehouse_id = databricks_warehouse_id
        self.DEBUG_MODE = False
        self.logger = self._setup_logger()
        self.load_context()  # Load context immediately

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_context(self):
        try:
            self.logger.info("Loading context...")
            llm_model = ChatDatabricks(endpoint=self.llm_model_name, max_tokens=3000, temperature=0.0)
            
            self.chain_query_classification = create_chain_query_classification(llm_model)
            self.code_regulation_rag_chain = create_code_regulation_rag_chain(
                llm_model, self.catalog_name, self.schema_name, self.databricks_url, self.token)
            self.room_identification_chain = create_room_identification_chain(llm_model)
            self.bim_revit_data_chain = create_bim_revit_data_chain(
                self.catalog_name, self.schema_name, llm_model, self.databricks_url, self.token, self.databricks_warehouse_id)
            self.compliance_check_chain = create_compliance_check_chain(llm_model)
            self.chain_general_response = create_general_response_chain(llm_model)
            self.logger.info("Context loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error in load_context: {str(e)}")
            raise

    def debug_print(self, message):
        if self.DEBUG_MODE:
            self.logger.debug(message)

    def debug_wrapper(self, runnable, step_name, timeout=180):
        def wrapper(x):
            start_time = time.time()
            self.debug_print(f"Debug - {step_name} input: {x}\n")
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(runnable, x)
                    result = future.result(timeout=timeout)
                self.debug_print(f"Debug - {step_name} output: {result}\n")
                return result
            except concurrent.futures.TimeoutError:
                error_msg = f"Timeout in {step_name} after {timeout} seconds\n"
                self.logger.error(error_msg)
                return {"error": error_msg}
            except Exception as e:
                error_msg = f"Error in {step_name}: {str(e)}\n"
                self.logger.error(error_msg)
                return {"error": error_msg}
            finally:
                elapsed_time = time.time() - start_time
                self.debug_print(f"Debug - {step_name} took {elapsed_time:.2f} seconds\n")
        return wrapper

    def multi_stage_chain(self, x):
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
        if 'debug_mode' in model_input.columns:
            self.DEBUG_MODE = bool(model_input['debug_mode'].iloc[0])
        else:
            self.DEBUG_MODE = False

        results = []
        for _, input_row in model_input.iterrows():
            input_item = input_row.to_dict()
            try:
                result = self.multi_stage_chain(input_item)
                if self.DEBUG_MODE:
                    self.logger.debug(f"Debug - Final output: {result}\n")
                result_json = json.dumps(result)
                results.append(result_json)
            except Exception as e:
                error_msg = f"Error in multi_stage_chain: {str(e)}"
                self.logger.error(error_msg)
                results.append(json.dumps({"error": error_msg}))
        return results