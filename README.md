<div align="center">
  <img src="https://github.com/user-attachments/assets/08b17550-558e-4138-9045-a16bf7eace05" width="450" alt="ReguBIM AI Logo">
</div>

ReguBEAM AI is a generative AI-powered compliance assistant that revolutionizes regulatory verification and building design compliance. By leveraging BIM models and regulatory documents, ReguBEAM AI streamlines building code checks, reducing time, costs, and human error.

## 📂 Project Structure

```
.
├── app/                    # ReguBIM AI deployment application
├── bim-revit-processing/   # Revit BIM data processing scripts
├── databricks-notebooks/   # Databricks workflow notebooks
```

For detailed information about each section and how to use it, please refer to their respective README files:
- [app/README.md](./app)
- [bim-revit-processing/README.md](./bim-revit-processing)
- [databricks-notebooks/README.md](./databricks-notebooks)

## 🏗️ System Architecture
ReguBIM AI is built on Databricks' scalable architecture, utilizing a medallion pattern for efficient data management. The system automates data processing through ETL pipelines triggered by file arrivals, transforming both BIM models and regulatory documents from raw data to analysis-ready formats. The application is deployed as a Databricks App, leveraging Databricks' serverless infrastructure.

<div align="center">
  <img src="https://github.com/user-attachments/assets/57f2142a-ef2f-49ef-8e38-39fee5f26d38" width="850" alt="Overall Process">
</div>

1. **Application**
   * Databricks App deployment
   * ReguBIM AI interface
   * Serverless compute utilization

2. **Data Infrastructure**
   * Unity Catalog (Tables & Volumes management)
   * Delta Lake storage format
   * Vector Search for similarity search
   * Serverless SQL warehouse

3. **Models**
   * Hugging Face transformers for embeddings
   * Llama 3.1 models for Multi-intent classification: 
     * Query Intent Classification
     * Building Codes & Regulations Analysis
     * BIM Data Spatial Analysis
     * Automated Compliance Checks
     * General Response Chain
   * MLflow for model management and versioning

4. **ETL Pipeline (Medallion Architecture)**
   * Bronze: Raw BIM JSON and regulatory documents
   * Silver: Spatial connections and regulatory transformations
   * Gold: Business-level aggregated data and embeddings index

5. **Automation & Compute**
   * File-triggered Databricks Jobs
   * Serverless compute provisioning
   * Automated scaling

6. **Security**
   * Databricks Secret Scope
   * Access token management
   * Secure credential storage
  
## 🌟 Features

ReguBIM AI processes user queries through different chains:
<div align="center">
  <img src="https://github.com/user-attachments/assets/f5dc03d6-e385-4b60-be5a-1fdd9b948bb4" width="650" alt="Overall Process">
</div>

1. **Building Codes & Regulations Analysis**
   <div align="left">
      <img src="https://github.com/user-attachments/assets/604f28a5-9470-4cde-adb8-a0e1b95d6cd6" width="650" alt="Code and Regulation RAG Chain">
   </div>
* Converts regulatory documents into searchable vector embeddings
* Uses transformer models for semantic understanding
* Uses RAG to analyze and retrieve relevant building codes
* Employs re-ranking to ensure most relevant regulations are retrieved
* Sample queries:
   * "What are the code and regulations for FCC?" 
   * "What are FCC Room Requirements I have to comply with?"

2. **BIM Data Spatial Analysis**
   <div align="left">
      <img src="https://github.com/user-attachments/assets/fb38cb10-3865-4937-82d1-7a3b2d4b6173" width="650" alt="Code and Regulation RAG Chain">
   </div>
* Uses ReAct agent for spatial analysis and graph-based exploration
* Performs pathfinding and spatial relationship analysis using NetworkX
* Creates Room Connectivity Graphs for spatial relationship analysis
* Calculates distances using Manhattan distance for realistic measurements
* Sample queries:
   * "How many rooms are there and can you list all the rooms?"
   * "What is the path from FCC to Staircase?"

3. **Automated Compliance Checks**
   <div align="left">
      <img src="https://github.com/user-attachments/assets/346078ae-abc4-4a10-9cb4-fc49a22c3967" width="650" alt="Code and Regulation RAG Chain">
   </div>
* Cross-references BIM data against regulations for compliance verification
* Integrates both regulatory requirements and spatial analysis
* Generates detailed compliance reports with recommendations
* Identifies specific violations and suggests corrections
* Sample queries:
   * "Does the FCC comply with codes and regulations?"

Additionally, ReguBIM AI includes a **General Response Chain** for handling basic interactions and queries:
* Processes general inquiries and conversational interactions
* Provides helpful guidance to users
* Example: "Hello, can you help me?"

### 💡 How Queries Are Processed

When you input a query, ReguBIM AI classifies and processes it through the appropriate chain. Here are some examples:

| Query | Classification | Explanation |
|-------|---------------|-------------|
| "Hello, can you help me?" | General Response Chain | Basic greeting and assistance request |
| "How many rooms are there and can you list all the rooms?" | BIM Revit Data Chain | Requests spatial information from BIM data |
| "What are FCC Room Requirements I have to comply with?" | Code and Regulation RAG Chain | Seeks regulatory information about FCC requirements |
| "What is the path from FCC to Staircase?" | BIM Revit Data Chain | Requests pathfinding analysis between spaces |
| "Does the FCC comply with code and regulation?" | Compliance Check Chain | Requests comprehensive compliance verification |
