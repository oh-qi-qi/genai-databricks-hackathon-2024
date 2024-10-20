from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import LLMChain, StuffDocumentsChain

import pandas as pd

import os
import re
import ast
import json
import logging
from typing import List, Dict, Any, Union

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

