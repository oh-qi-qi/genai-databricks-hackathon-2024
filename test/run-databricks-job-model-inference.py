from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv
import requests
import time
import json
import os

# Load environment variables from the .env file
load_dotenv()

# Access the variables
DATABRICKS_URL = os.getenv('DATABRICKS_URL')
TOKEN = os.getenv('DATABRICKS_TOKEN')
JOB_ID = os.getenv('DATABRICKS_JOB_ID')

# Function to trigger the job with notebook parameters
def trigger_job(query_text, debug_mode):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    payload = {
        "job_id": JOB_ID,
        "notebook_params": {
            "query_text": query_text,  # The text query
            "debug_mode": "True" if debug_mode else "False"  # Debug mode as string "True" or "False"
        }
    }
    response = requests.post(f'{DATABRICKS_URL}/api/2.1/jobs/run-now', headers=headers, json=payload)
    response.raise_for_status()  # Raise an error if request fails
    return response.json()['run_id']

# Function to poll job status until completion
def wait_for_job_completion(run_id):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    while True:
        response = requests.get(f'{DATABRICKS_URL}/api/2.1/jobs/runs/get?run_id={run_id}', headers=headers)
        response.raise_for_status()
        job_status = response.json()['state']['life_cycle_state']
        
        if job_status == 'TERMINATED':
            print("Job completed!")
            break
        elif job_status == 'INTERNAL_ERROR' or job_status == 'SKIPPED':
            raise Exception(f"Job failed with status: {job_status}")
        else:
            print(f"Job is {job_status}. Waiting...")
            time.sleep(10)  # Poll every 10 seconds

# Function to retrieve job output
def get_job_output(task_run_id):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    response = requests.get(f'{DATABRICKS_URL}/api/2.1/jobs/runs/get-output?run_id={task_run_id}', headers=headers)
    
    response.raise_for_status()
    
    return response.json()['notebook_output']['result']

def get_task_run_id(job_run_id):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    response = requests.get(f'{DATABRICKS_URL}/api/2.1/jobs/runs/get?run_id={job_run_id}', headers=headers)
    task_run_id = 0
    
    if response.status_code == 200:
        run_details = response.json()
        
        tasks = run_details['tasks']
        
        for task in tasks:
            task_run_id = task['run_id']
            print(f"Task: {task['task_key']}, Task Run ID: {task_run_id}")
            
    else:
        print(f"Failed to get run details: {response.status_code}, {response.text}")
        
    return task_run_id

def print_nested_dict_display(data):
    console = Console()

    def format_value(value):
        if isinstance(value, str):
            try:
                # Try to parse as JSON first
                json_data = json.loads(value)
                return Markdown(f"```json\n{json.dumps(json_data, indent=2)}\n```")
            except json.JSONDecodeError:
                # If not JSON, treat as Markdown
                return Markdown(value)
        elif isinstance(value, dict):
            return Markdown(f"```json\n{json.dumps(value, indent=2)}\n```")
        else:
            return str(value)

    for key, value in data.items():
        formatted_value = format_value(value)
        panel = Panel(formatted_value, title=key, expand=False)
        console.print(panel)

# Main function
if __name__ == "__main__":
    try:
        # Trigger the job with parameters
        query_text = "Does the FCC comply with code and regulation?"
        debug_mode = False  # or True based on your preference

        run_id = trigger_job(query_text, debug_mode)
        print(f"Job triggered. Run ID: {run_id}")

        # Wait for the job to complete
        wait_for_job_completion(run_id)
        task_run_id = get_task_run_id(run_id)
    
        # Get the job output
        print(f"Retrieving job task output. Task run ID: {task_run_id}")
        result = get_job_output(task_run_id)
        
        json_output = json.loads(result)
        
        print_nested_dict_display(json_output)
        
    except Exception as e:
        print(f"Error: {e}")