## Overview
This application provides chat history analysis capabilities with integration to Databricks. For this Hackathon it only focused on Extra Low Voltage section and Fire Command Centre (FCC).

![image](https://github.com/user-attachments/assets/321bfc2d-0ce1-4d62-a057-5c323b3cab1f)


## Folder Structure
```
app/
├── .databricks/                    # Databricks configuration files
├── main/
│   ├── __pycache__/               # Python cache directory
│   ├── sample_data/               # Sample data used for the project
│   ├── assets/                    # Static assets (e.g logo)
│   ├── common/                    # Shared utilities and helper functions
│   ├── databricks_scripts/        # SQL and Python scripts for Databricks integration
│   └── visualization_templates/   # HTML Templates for generating visual responses
├── __init__.py                 
├── app.py                         # Main Streamlit application entry point
├── app.yml                        # Application configuration settings
├── chat_history.json              # Log of user interactions and queries
├── .gitignore                     # Git ignore rules
├── possible_questions.txt         # Sample questions
└── requirements.txt               # Python package dependencies
```
## Setup and Configuration

### Environment Variables
Create a `.env` file in the root directory with the following parameters:
```
DATABRICKS_HOST=<your-databricks-host>
DATABRICKS_TOKEN=<your-databricks-token>
DATABRICKS_WAREHOUSE_ID=<your-warehouse-id>
```

### Databricks Secret Scope Setup
1. Navigate to your Databricks workspace
2. Set up a secret scope with the following secrets:
   - DATABRICKS_HOST
   - DATABRICKS_TOKEN
   - DATABRICKS_WAREHOUSE_ID

## Running the Application

### Method 1: Local Development
1. Create a conda environment
   ```bash
   conda create -n test_env python=3.10.12
   conda activate test_env
   ```
2. Clone the repository
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Navigate to the main folder
5. Run the application:
   ```bash
   streamlit run main/app.py
   ```

### Method 2: Databricks App
The application can be deployed via Databricks App, providing seamless integration with your Databricks workspace and native access to workspace resources. Some benefits are:
Seamless workspace integration
Automatic credential management
Built-in security controls
Real-time data access
Sample URL Link: https://regubimai-1294267129871822.aws.databricksapps.com/
For comprehensive information about Databricks Apps, including deployment, configuration, and best practices, please refer to the official [Databricks Apps](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html)documentation.

## Prerequisites
- Python 3.10.12
- Streamlit
- Databricks account with appropriate permissions
- Access to Databricks workspace (for Method 2)

## Notes
- Ensure all environment variables are properly set before running the application
- For local development, verify that your Databricks credentials are correctly configured
- The application requires proper network connectivity to access Databricks services

## Sample Questions and Features
The application is designed to handle queries such as:

1. Basic assistance:
   ```
   Q: Hello, can you help me?
   A: Provides overview of available assistance
   ```
   ![image](https://github.com/user-attachments/assets/3c5179f5-9889-439e-a496-c88a565305d9)

2. Room information:
   ```
   Q: How many rooms are there and can you list all the rooms?
   A: Returns total room count and categorized list of all rooms
   ```
   ![image](https://github.com/user-attachments/assets/abad1368-9a08-4c9c-992a-4bbf951b8966)

3. Compliance queries:
   ```
   Q: What are FCC Room Requirements I have to comply with?
   A: Lists all relevant FCC compliance requirements
   ```
   ![image](https://github.com/user-attachments/assets/458f3494-2637-43b3-9cd3-b1cdb9538280)

4. Navigation assistance:
   ```
   Q: What is the path from FCC to Staircase?
   A: Generates available paths and visual representation if applicable
   ```
   ![image](https://github.com/user-attachments/assets/817b9191-4349-4971-b5e7-6a1f41b6fffd)

5. Compliance verification:
   ```
   Q: Does the FCC comply with code and regulation?
   A: Provides compliance status with detailed breakdown
   ```
   ![image](https://github.com/user-attachments/assets/1d2c60db-d68a-4ba3-bfaa-215708240ed6)

## Support
For any issues or questions, please contact the development team or refer to the internal documentation.
