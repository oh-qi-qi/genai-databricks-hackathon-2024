# ReguBIM AI

ReguBIM AI tackles the intricate challenge of handling building code regulations, BIM data analysis, and compliance checking in the construction and engineering domain. It is made possible by leveraging Databricks' comprehensive suite of tools for scalable, efficient, and secure processing of regulatory and BIM data.

## 🌟 Features

- **Unified Data Management**: Efficient data governance via Unity Catalog
- **AI Capabilities**: Leverages cutting-edge NLP for advanced query handling
- **Scalable Processing**: Apache Spark and Delta Lake for distributed data handling
- **ML Ops Integration**: MLflow for streamlined machine learning operations
- **Flexible Analysis**: Combination of graph processing, vector search, and SQL

## 📂 Project Structure

```
.
├── app/                    # ReguBIM AI deployment application
├── bim-revit-processing/   # Revit BIM data processing scripts
├── databricks-notebooks/   # Databricks workflow notebooks
└── test/                  # Test suite
```

For detailed information about each component, please refer to their respective README files:
- [app/README.md](./app/README.md)
- [bim-revit-processing/README.md](./bim-revit-processing/README.md)
- [databricks-notebooks/README.md](./databricks-notebooks/README.md)

## 🏗️ System Architecture

### Core Components

1. **Query Classification**
   - Type: LLM Model (Prompt Chaining)
   - Technology: Databricks-hosted LLM (Meta's Llama 3.1)
   - Categories:
     - Building Codes and Regulations
     - BIM Revit Data
     - Compliance Check
     - General queries

2. **Code Regulation RAG Chain**
   - Vector Database (Databricks Vector Search)
   - Embedding Model (Sentence Transformers)
   - Document Retrieval
   - Context Creation
   - LLM Model for RAG

3. **BIM Revit Data Chain**
   - Spark Tables (Edges and Vertices)
   - Graph Processing with NetworkX:
     - Room connectivity analysis
     - Path finding and optimization
     - Spatial relationship mapping
   - Table Generation
   - LLM Model (ReAct Agent)

4. **Compliance Check Full Chain**
   - Code Regulation RAG Chain
   - Room Identification Chain
   - BIM Revit Data Chain
   - Compliance Assessment

5. **General Response Chain**
   - Handles general queries via LLM prompt chaining


## 🛠️ Technical Stack

### Core Technologies
- **LLM**: Databricks-hosted Meta's Llama 3.1
- **Vector Search**: Databricks Vector Search
- **Embeddings**: Sentence Transformers
- **Data Processing**: Apache Spark SQL
- **Graph Processing**: NetworkX
  - Data structures for graphs, digraphs, and multigraphs
  - Standard graph algorithms for path finding and analysis
- **ML Lifecycle**: MLflow
- **LLM Orchestration**: LangChain


### Key Benefits
- Modular and scalable architecture
- Distributed computing capabilities
- Integration of cutting-edge NLP with domain-specific processing
- Comprehensive building code compliance solution
- Robust graph analysis through NetworkX
