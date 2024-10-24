## Project Overview
This project contains a collection of Databricks notebooks for processing and analyzing code regulations, Building Information Modeling (BIM) data and compliance requirements.

## Project Structure

```
project_root/
│
├── bim-revit/
│   ├── revit_bim_data_processing.ipynb
│   ├── revit_bim_explore_data.ipynb
│   └── revit_bim_react_agent.ipynb
│
├── code-and-regulations/
│   ├── code_regulation_data_processing.ipynb
│   ├── code_regulation_embedding_model.ipynb
│   └── code_regulation_rag_chain.ipynb
│
├── common/
│   └── installation_setup.ipynb
│
├── compliance/
│   └── regulation-bim-compliance.ipynb
│
├── intent-category/
│   └── query_intent_chain.ipynb
│
├── databricks_base_environment.yml
└── requirements.txt
```

## Detailed Notebook Descriptions

### 📁 bim-revit/
Notebooks focused on BIM and Revit data processing:

- **revit_bim_data_processing.ipynb**
  - Primary data processing pipeline for Revit/BIM json format files
  - Performs data cleaning and standardization
  - Creates structured datasets for analysis

- **revit_bim_explore_data.ipynb**
  - Exploratory data analysis of processed BIM data
  - Generates visualizations of building components

- **revit_bim_react_agent.ipynb**
  - Interactive agent for automated BIM analysis
  - Processes user queries about BIM models
  - Generates automated responses and recommendations

### 📁 code-and-regulations/
Notebooks for building code analysis and regulation processing:

- **code_regulation_data_processing.ipynb**
  - Processes raw building codes and regulations
  - Extracts relevant clauses and requirements

- **code_regulation_embedding_model.ipynb**
  - Creates vector representations of building codes
  - Enables semantic search of regulations

- **code_regulation_rag_chain.ipynb**
  - Implements Retrieval-Augmented Generation (RAG) for regulations
  - Provides context-aware regulatory responses

### 📁 common/
Shared utilities and setup:

- **installation_setup.ipynb**
  - Configures Databricks environment
  - Sets up connection to necessary services

### 📁 compliance/
Compliance verification tools:

- **regulation-bim-compliance.ipynb**
  - Automated compliance checking system
  - Matches BIM model elements against regulations
  - Generates compliance reports
  - Identifies potential violations and issues
  - Provides recommendations for compliance

### 📁 intent-category/
Classification and query processing:

- **query_intent_chain.ipynb**
  - Processes and categorizes user queries
  - Identifies intent behind questions
  - Routes queries to appropriate chain

### Configuration Files

- **databricks_base_environment.yml**
  - Defines Databricks configuration to install dependencies

- **requirements.txt**
  - Lists all Python package dependencies
  - Specifies version requirements

## Environment Setup
All required Python packages are listed in `requirements.txt`. The project uses a Databricks environment as specified in `databricks_base_environment.yml`.
1. Create a new Databricks cluster
2. If using servless compute, install [Notebook Dependencies](https://docs.databricks.com/en/compute/serverless/dependencies.html) with the configuration in `databricks_base_environment.yml`

## Note
- All notebooks are designed to run in Databricks environment
- Ensure proper access permissions are set up in your Databricks workspace
