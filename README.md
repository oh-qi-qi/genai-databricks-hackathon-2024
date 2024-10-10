### Overview

ReguBIM  tackles the intricate challenge of handling building code regulations, BIM data analysis, and compliance checking in the construction and engineering domain. It is made possible by leveraging Databricks' comprehensive suite of tools for scalable, efficient, and secure processing of regulatory and BIM data. It integrates advanced LLM capabilities with traditional data processing, ensuring a powerful solution for complex compliance checks in the construction industry.This system can seamlessly scale and adapt to growing complexity in building code compliance and BIM data analysis.

Key Components include:
- **Unified Data**: Efficient data governance via Unity Catalog.
- **AI Capabilities**: Leverages cutting-edge NLP for advanced query handling.
- **Scalable Processing**: Apache Spark and Delta Lake for distributed data handling.
- **ML Ops Integration**: MLflow for streamlined machine learning operations.
- **Flexible Analysis**: Combination of graph processing, vector search, and SQL.

Below are the key takeaways:

1. **System Components:**
   - **Query Classification**: LLM-based prompt chaining to categorize user queries.
   - **Code Regulation RAG Chain**: Retrieval-Augmented Generation for answering regulatory queries.
   - **BIM Revit Data Chain**: Graph-based analysis of Building Information Modeling (BIM) data.
   - **Compliance Check Full Chain**: Integrates multiple sub-chains for comprehensive compliance assessment.
   - **General Response Chain**: Handles non-specific queries.

2. **Key Technologies:**
   - **Databricks-hosted LLM (Meta's Llama 3.1)**: Core of the natural language processing and generation tasks.
   - **Databricks Vector Search**: Efficient storage and retrieval of document embeddings.
   - **Apache Spark SQL & GraphFrames**: Processes BIM data as a graph structure.
   - **Unity Catalog**: Manages data centrally for governance and scalability.
   - **MLflow**: Manages the machine learning lifecycle.
   - **LangChain**: Supports LLM-based workflows and agents.

3. **Data Management:**
   - Unity Catalog integrates five key tables for code regulations and BIM data, ensuring centralized management and scalability.

4. **Unique Features:**
   - Combines traditional data processing with advanced LLM-based analysis.
   - Utilizes graph-based BIM representations for spatial queries.
   - Employs RAG and ReAct methodologies for enhanced LLM performance.

5. **System Strengths:**
   - Modular and scalable, allowing for easy updates and maintenance.
   - Leverages distributed computing on Databricks for efficient large-scale processing.
   - Integrates cutting-edge NLP techniques with domain-specific data processing.
   - Provides a comprehensive solution for building code compliance and BIM data analysis.
---

## System Overview
![system-overview](https://github.com/user-attachments/assets/11e2815f-039c-49f7-9dbc-52fd807f047b)

### 1. Query Classification
- **Type**: LLM Model (Prompt Chaining)
- **Technology**: Databricks-hosted LLM (Meta's Llama 3.1)
- **Function**: Classifies user queries into four categories: 
  1. Building Codes and Regulations 
  2. BIM Revit Data 
  3. Compliance Check 
  4. General (other)

### 2. Code Regulation RAG Chain
- **Purpose**: Handles regulatory queries by using Retrieval-Augmented Generation (RAG).
- **Components**:
  #### a. Vector Database
  - **Technology**: Databricks Vector Search
  - **Function**: Stores and indexes document embeddings for fast retrieval.
  - **Tables**:
    - `code_regulations_engineering_chunk_embedding`
    - `code_regulations_engineering_self_managed_vs_index`
    - `code_regulations_engineering_pdf_raw_text`
  
  #### b. Embedding Model
  - **Technology**: Sentence Transformers
  - **Function**: Converts text into vector embeddings for semantic search.
  
  #### c. Document Retrieval
  - **Function**: Fetches relevant documents based on query embeddings.
  
  #### d. Context Creation
  - **Function**: Creates a context using the retrieved documents for the LLM.
  
  #### e. LLM Model (RAG)
  - **Technology**: Databricks-hosted LLM (Meta's Llama 3.1)
  - **Function**: Generates answers based on the retrieved context.

### 3. BIM Revit Data Chain
- **Purpose**: Analyzes BIM data using graph-based methodologies.
- **Components**:
  #### a. Spark Tables (Edges and Vertices)
  - **Technology**: Databricks Spark SQL
  - **Tables**:
    - `revit_room_edges`: Room connections.
    - `revit_room_vertices`: Individual rooms.
  
  #### b. Data Retrieval
  - **Function**: Fetches relevant BIM data from Unity Catalog.
  
  #### c. Room Path Calculation
  - **Technology**: GraphFrames
  - **Function**: Calculates paths between rooms using graph algorithms.
  
  #### d. Room Relationship Analysis
  - **Technology**: GraphFrames
  - **Function**: Analyzes relationships between rooms (e.g., adjacency, connectivity).
  
  #### e. Table Generation
  - **Function**: Formats the retrieved data into structured tables for analysis.
  
  #### f. LLM Model (ReAct Agent)
  - **Technology**: Databricks-hosted LLM (Meta's Llama 3.1)
  - **Function**: Uses the ReAct methodology to generate responses based on BIM data.

### 4. Compliance Check Full Chain
- **Purpose**: Conducts end-to-end compliance assessments.
- **Sub-Chains**:
  #### a. Code Regulation RAG Chain
  - Reuses the regulation chain for retrieving code-related information.
  
  #### b. Room Identification Chain
  - **Technology**: LLM (Meta's Llama 3.1)
  - **Function**: Extracts and identifies room information from regulatory text.
  
  #### c. BIM Revit Data Chain
  - Reuses the BIM data chain for room and pathfinding analysis.
  
  #### d. Compliance Check Chain
  - **Function**: Assesses compliance using all gathered data, providing justifications and recommendations.

### 5. General Response Chain
- **Type**: LLM Model (Prompt Chaining)
- **Technology**: Databricks-hosted LLM (Meta's Llama 3.1)
- **Function**: Handles general queries that don't fit the other categories.

---

### Unity Catalog Integration
Unity Catalog plays a key role in managing the system’s data:
1. **Tables**:
   - `code_regulations_engineering_chunk_embedding`
   - `code_regulations_engineering_pdf_raw_text`
   - `code_regulations_engineering_self_managed_vs_index`
   - `revit_room_edges`
   - `revit_room_vertices`
   
2. **Key Benefits**:
   - Centralized data management.
   - Fine-grained access control.
   - Data scalability and governance.

---

### Core Packages and Dependencies Breakdown

#### 1. Databricks-hosted LLM (Meta's Llama 3.1)
- **Type**: Large Language Model
- **Usage**: Core of query classification, RAG, ReAct, and compliance assessment.

#### 2. Databricks Vector Search
- **Type**: Vector Database
- **Usage**: Stores document embeddings for regulatory queries.

#### 3. Sentence Transformers
- **Type**: Embeddings Library
- **Usage**: Generates vector embeddings for regulatory texts.

#### 4. Apache Spark SQL
- **Type**: Distributed SQL Engine
- **Usage**: Queries and processes BIM data stored in Unity Catalog tables.

#### 5. GraphFrames
- **Type**: Graph Processing Library
- **Usage**: Processes BIM data to calculate room paths and relationships.

#### 6. MLflow
- **Type**: Machine Learning Lifecycle Manager
- **Usage**: Manages the system’s lifecycle, versioning, and deployment.

#### 7. LangChain
- **Type**: LLM Workflow Framework
- **Usage**: Supports prompt chaining and agent workflows in the system.
