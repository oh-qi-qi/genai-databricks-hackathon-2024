from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain


def create_general_response_chain(llm_model):
    general_prompt = PromptTemplate.from_template("""
    You are ReguBIM AI, a GenAI-powered compliance assistant that simplifies the building compliance process. You integrate graph analysis of BIM data with building codes and regulations, enabling users to query both regulatory requirements and BIM data effortlessly, streamlining the compliance process for BIM models.

    You are an assistant for a building information system. The user has asked a question that doesn't fit into our specific categories of building codes, BIM data, or compliance checks. Please provide a helpful and friendly response.
    User query: {query}
    Response:
    """)

    chain = LLMChain(llm=llm_model, prompt=general_prompt)

    def process_query(input_dict):
        query = input_dict.get('query', '')
        result = chain.invoke({"query": query})
        return {
            "input": query,
            "output": result["text"]
        }

    return RunnableLambda(process_query)