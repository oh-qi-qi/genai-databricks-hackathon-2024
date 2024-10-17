# Databricks notebook source
# MAGIC %md
# MAGIC #### Dependency and Installation

# COMMAND ----------

# MAGIC %run ../common/installation_setup

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Code and Regulation Rag Chain 
# MAGIC - Retriever
# MAGIC - Prompt Template
# MAGIC - LLM Chain

# COMMAND ----------

# Check the registered models
import mlflow

print(mlflow.__version__)

from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Search for all registered models
registered_models = client.search_registered_models()
embedding_model = ""
embedding_model_name = ""
embedding_model_full_name = ""
embedding_model_version = 1

# Print details of each registered model
for model in registered_models:
    if (f"{catalog_name}.{schema_name}") in model.name:

        embedding_model_full_name = model.name
        embedding_model_name = model.name.replace(f"{catalog_name}.{schema_name}.", "")
        print(f"Name: {embedding_model_name}")

        embedding_model = client.get_registered_model(name=model.name)
        versions = client.search_model_versions(filter_string=f"name='{model.name}'")

        for version in versions:
            print(f" - Version: {version.version}, Stage: {version.current_stage}")
        
        # Get the latest version    
        embedding_model_version= versions[-1].version

run_id = client.get_model_version(name=embedding_model_full_name, version=embedding_model_version).run_id

try:
    loaded_embedding_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{embedding_model_name}")
    print(f"Model {embedding_model_name} loaded successfully from run ID")
except Exception as e:
    print(f"Error loading from run ID: {str(e)}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.embeddings import DatabricksEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.docstore.document import Document
from flashrank import Ranker, RerankRequest

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda

import mlflow.pyfunc
import pandas as pd

def create_code_regulation_rag_chain(llm_model):
    vs_endpoint_name = f"vs_endpoint_{catalog_name}"
    vs_index_fullname = f"{catalog_name}.{schema_name}.code_regulations_engineering_self_managed_vs_index"

    def retrieve(inputs):
        query = inputs
        vsc = VectorSearchClient(disable_notice=True)
        vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)
        query_embeddings = loaded_embedding_model.predict(pd.DataFrame({"text": [query]}))
        query_vector = query_embeddings.iloc[0].tolist()
        retrieved = vs_index.similarity_search(
            query_vector=query_vector,
            columns=["id", "document_name", "section", "content", "url"],
            num_results=10)
        return {'query': query, 'retrieved': retrieved}

    def rerank(inputs, ranker_model="ms-marco-MiniLM-L-12-v2", k=10):
        query = inputs['query']
        retrieved = inputs['retrieved']
        passages = [
            {
                "score": doc[-1], 
                "id": doc[0], 
                "file": doc[1], 
                "section": doc[2], 
                "chunk": doc[3],
                "url": doc[4]
            } for doc in retrieved.get("result", {}).get("data_array", [])
        ]
        ranker = Ranker(model_name=ranker_model)
        rerankrequest = RerankRequest(query=query, passages=[
            {
                'id': passage['id'],
                'text': passage['chunk'],
                "meta": {
                    "section": passage['section'],
                    "document_name": passage['file'],
                    "url": passage['url']
                }
            } for passage in passages
        ])
        results = ranker.rerank(rerankrequest)[:k]
        reranked_docs = [Document(
            page_content=r.get("text"),
            metadata={
                "rank_score": r.get("score"), 
                "vector_id": r.get("id"),
                "section": r.get("meta").get("section"),
                "document_name": r.get("meta").get("document_name"),
                "url": r.get("meta").get("url")
            }
        ) for r in results]
        return {'query': query, 'retrieved_docs': reranked_docs}

    def process_retrieved_docs(inputs):
        query = inputs['query']
        docs = inputs['retrieved_docs']
        grouped_documents = {}
        metadata_section = []
        
        for doc in docs:
            key = f"{doc.metadata.get('document_name')} - {doc.metadata.get('section')}"
            if key not in grouped_documents:
                grouped_documents[key] = []
            grouped_documents[key].append(doc.page_content)
            
            if key not in metadata_section:
                metadata_info = f"Section: {key}"
                url = doc.metadata.get('url')
                if url:
                    metadata_info += f", URL: {url}"
                metadata_section.append(metadata_info)

        final_context = "\n\n".join("\n".join(pages) for pages in grouped_documents.values())
        final_metadata_section = ", ".join(metadata_section)

        return {
            "context": final_context,
            "metadata_section": final_metadata_section,
            "input": query,
            "input_documents": docs
        }

    def format_output(inputs):
        output_text = inputs.get('output_text', '')
        input_documents = inputs.get('input_documents', [])
        
        references_dict = {}
        
        for doc in input_documents:
            doc_name = doc.metadata.get('document_name', 'Unknown Document')
            full_section = doc.metadata.get('section', 'Unknown Section')
            url = doc.metadata.get('url', '')
            
            parts = full_section.split(',', 1)
            main_section = parts[0].strip()
            subsection = parts[1].strip() if len(parts) > 1 else ''
            
            if doc_name not in references_dict:
                references_dict[doc_name] = {'url': url, 'main_section': main_section, 'subsections': set()}
            
            if subsection:
                references_dict[doc_name]['subsections'].add(subsection)
        
        references = []
        for doc_name, info in references_dict.items():
            url = info['url']
            main_section = info['main_section']
            subsections = ', '.join(sorted(info['subsections']))
            
            if subsections:
                section_text = f"{main_section}, {subsections}"
            else:
                section_text = main_section
            
            if url:
                reference = f"* [{doc_name}]({url}), Section: {section_text}"
            else:
                reference = f"* {doc_name}, Section: {section_text}"
            
            references.append(reference)
        
        if references:
            references_text = "\n".join(sorted(references))
            output_text += f"\n\nTo ensure compliance, please refer to the following documents:\n\n{references_text}"
        
        return {'input': inputs.get('input'), 'output': output_text}
    
    # Define the main prompt template
    code_regulation_prompt_template = """
    You are an assistant specializing in building codes, safety regulations, and design standards for various room types in buildings. Your task is to extract and provide relevant information about codes, regulations, and standards for the room type mentioned in the question.

    Use the following pieces of context and metadata:

    <context>
    {context}
    </context>

    <metadata>
    Section: {metadata_section}
    </metadata>

    Follow these steps:
    1. Identify the room type mentioned in the question.
    2. Extract and list all relevant codes, regulations, standards, and requirements for the identified room type. Include specific measurements, materials, equipment, location, and any other pertinent details if available.
    3. Organize the information clearly, grouping related requirements together.
    4. If specific information for the mentioned room type is not available, provide general building codes or regulations that might be applicable.

    Provide only factual information from the given context. Do not make assumptions or assessments about compliance. If certain information is not available, clearly state this.

    Question: {input}

    Answer:
    """

    # Create the main prompt
    code_regulation_prompt = ChatPromptTemplate.from_template(code_regulation_prompt_template)

    # Create the document prompt
    code_regulation_prompt_document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm_model, prompt=code_regulation_prompt)

    # Create the StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=code_regulation_prompt_document_prompt,
        document_variable_name="context"
    )

    # Combine the chains
    retriever_chain = RunnableLambda(retrieve) | RunnableLambda(rerank)
    final_chain = retriever_chain | RunnableLambda(process_retrieved_docs) | stuff_chain | RunnableLambda(format_output)

    return final_chain

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

llm_model = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=3000, temperature=0.0)
code_regulation_rag_chain = create_code_regulation_rag_chain(llm_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the code and regulation rag chain

# COMMAND ----------

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Example question to pass to your process_question function
code_regulation_question_1 = {"query": "What are the regulations for fire exits?"}

# Call the function to get the answer
code_regulation_answer_1 = code_regulation_rag_chain.invoke(code_regulation_question_1["query"])

console = Console()

# Create Markdown objects with added headers
md_input_1 = Markdown(f"**Input:**\n\n{code_regulation_answer_1['input']}")
md_output_1 = Markdown(f"**Output:**\n\n{code_regulation_answer_1['output']}")

# Print input and output in panels
console.print(Panel(md_input_1, title="Input", expand=False))
console.print(Panel(md_output_1, title="Output", expand=False))

# COMMAND ----------

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Example question to pass to your process_question function
code_regulation_question_2 = {"query": "What are the regulations for fcc?"}

# Call the function to get the answer
code_regulation_answer_2 = code_regulation_rag_chain.invoke(code_regulation_question_2["query"])

console = Console()

# Create Markdown objects with added headers
md_input_2 = Markdown(f"**Input:**\n\n{code_regulation_answer_2['input']}")
md_output_2 = Markdown(f"**Output:**\n\n{code_regulation_answer_2['output']}")

# Print input and output in panels
console.print(Panel(md_input_2, title="Input", expand=False))
console.print(Panel(md_output_2, title="Output", expand=False))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Code and Regulation Rag Model and save to Registry (Optional)

# COMMAND ----------

# # Only run this if want to create model in model registry
# from mlflow.models import infer_signature
# import mlflow
# import langchain

# code_regulation_model_name = "code_regulations_rag_model"

# # save the langchain model to registry if needed
# with mlflow.start_run(run_name="code_regulations_rag_model") as run:
#     # Infer the signature from the question and answer
#     code_regulation_signature = infer_signature(code_regulation_question, code_regulation_answer)

#     # Log the code_regulation_final_chain model to MLflow with retriever
#     model_info = mlflow.langchain.log_model(
#         code_regulation_final_chain,
#         loader_fn=code_regulation_get_retriever,  # Ensure get_retriever is defined
#         artifact_path="code_regulation_final_chain",
#         registered_model_name=code_regulation_model_name,
#         pip_requirements=[
#             "mlflow==" + mlflow.__version__,
#             "langchain==" + langchain.__version__,
#             "databricks-vectorsearch",
#         ],
#         input_example=code_regulation_question,  # Set the input example as the question
#         signature=code_regulation_signature  # Add the inferred signature
#     )
