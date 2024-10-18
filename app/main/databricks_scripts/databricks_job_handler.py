from dotenv import load_dotenv
import requests
import json
import os
import time

# Load environment variables from the .env file
load_dotenv()

# Access the variables
DATABRICKS_URL = os.getenv('DATABRICKS_URL')
TOKEN = os.getenv('DATABRICKS_TOKEN')
JOB_ID = os.getenv('DATABRICKS_JOB_ID')

# Trigger the job with parameters
def trigger_databricks_job(query_text, debug_mode=False):
    query = str(query_text)
    print(f"Query: {query}")
    headers = {'Authorization': f'Bearer {TOKEN}'}
    payload = {
        "job_id": JOB_ID,
        "notebook_params": {
            "query_text": query,
            "debug_mode": "True" if debug_mode else "False"
        }
    }
    response = requests.post(f'{DATABRICKS_URL}/api/2.1/jobs/run-now', headers=headers, json=payload)
    response.raise_for_status()  # Raise an error if request fails
    return response.json()['run_id']

# Poll job status
def wait_for_job_completion(run_id):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    while True:
        response = requests.get(f'{DATABRICKS_URL}/api/2.1/jobs/runs/get?run_id={run_id}', headers=headers)
        response.raise_for_status()
        job_status = response.json()['state']['life_cycle_state']
        
        if job_status == 'TERMINATED':
            break
        elif job_status == 'INTERNAL_ERROR' or job_status == 'SKIPPED':
            raise Exception(f"Job failed with status: {job_status}")
        else:
            print(f"Job is {job_status}. Waiting...")
            time.sleep(10)  # Poll every 10 seconds

# Retrieve job output
def get_databricks_job_output(run_id):
    headers = {'Authorization': f'Bearer {TOKEN}'}
    response = requests.get(f'{DATABRICKS_URL}/api/2.1/jobs/runs/get-output?run_id={run_id}', headers=headers)
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
    else:
        print(f"Failed to get run details: {response.status_code}, {response.text}")
    return task_run_id