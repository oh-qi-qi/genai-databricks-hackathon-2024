from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import LLMChain, StuffDocumentsChain

import pandas as pd

import os
import sys
import json

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