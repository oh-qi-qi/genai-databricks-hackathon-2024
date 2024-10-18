from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatDatabricks
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.schema.runnable import RunnableMap, RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import langchain
import pandas as pd


def create_chain_query_classification(llm_model):
    # Query intent categories
    query_intent_categories = [
        "Building Codes and Regulations",
        "BIM Revit Data",
        "Compliance Check",
        "Other"
    ]

    # Intent query template
    query_intent_category_template = """
    You are an AI assistant tasked with categorizing user queries related to building codes, regulations, architectural data, and BIM (Building Information Modeling) elements from Revit. 
    Given the following categories:

    {categories}

    Classify the following query into one of these categories. If the query doesn't fit any category, classify it as "Other".
    Use the following guidelines:

    1. "Building Codes and Regulations": Queries about specific building codes, regulations, standards, room types (e.g., ELV rooms), disciplines (e.g., ELV, Electrical, Mechanical), or regulatory requirements. This includes questions about which rooms belong to or are managed by specific disciplines.

    2. "BIM Revit Data": Queries about physical characteristics of the building such as room id, sizes, locations, boundaries, room relationships, adjacencies, or counts of generic room types. This includes any spatial or structural data typically found in a Revit model. It does not include any information about the discipline that owns or manages which room, nor any regulatory or standard-based information.

    3. "Compliance Check": Queries that explicitly ask about how or whether the room complies with regulations or standards.

    4. "Other": Queries that don't fit into the above categories.

    Respond with only the category name, nothing else.

    User Query: {query}

    Category:"""

    query_intent_category_prompt = PromptTemplate(
        input_variables=["categories", "query"],
        template=query_intent_category_template
    )

    # Create the classification chain
    inner_chain = (
        RunnablePassthrough.assign(categories=lambda _: "\n".join(query_intent_categories))
        | query_intent_category_prompt
        | llm_model
        | StrOutputParser()
    )

    # Wrapper function to include input and output
    def chain_with_io(inputs):
        result = inner_chain.invoke(inputs)
        return {
            'input': inputs['query'],
            'output': result
        }

    # Convert the wrapper function to a RunnableLambda
    return RunnableLambda(chain_with_io)

# Test intent chain Sample

# Initialize the LLM
query_intent_category_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=500)

# Example input for signature inference (as pandas DataFrame)
chain_query_classification = create_chain_query_classification(query_intent_category_model)
intent_result = chain_query_classification.invoke({"query": "What are the building codes for ELV rooms?"})
print(intent_result)