import requests
import pandas as pd
import json
import sys
import os
from typing import List, Dict, Any

class DataRetrievalTool:
    def __init__(self, databricks_url: str, token: str, warehouse_id: str, catalog: str, schema: str):
        self.databricks_url = databricks_url.rstrip('/')
        self.token = token
        self.warehouse_id = warehouse_id
        self.catalog = catalog
        self.schema = schema
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        endpoint = f"{self.databricks_url}/api/2.0/sql/statements"
        data = {
            "statement": sql,
            "warehouse_id": self.warehouse_id,
            "catalog": self.catalog,
            "schema": self.schema,
        }
        
        response = requests.post(endpoint, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        while result['status']['state'] in ['PENDING', 'RUNNING']:
            statement_id = result['statement_id']
            result_url = f"{self.databricks_url}/api/2.0/sql/statements/{statement_id}"
            response = requests.get(result_url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
        
        if result['status']['state'] != 'SUCCEEDED':
            raise Exception(f"Query failed: {result['status'].get('error', {}).get('message', 'Unknown error')}")
        
        columns = [field['name'] for field in result['manifest']['schema']['columns']]
        data = result['result']['data_array']
        
        # Process the data, handling the 'bounds' column specially
        processed_data = []
        for row in data:
            processed_row = []
            for value in row:
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        processed_value = json.loads(value)
                    except json.JSONDecodeError:
                        processed_value = value
                else:
                    processed_value = value
                processed_row.append(processed_value)
            processed_data.append(processed_row)
        
        return pd.DataFrame(processed_data, columns=columns)

    def get_room_vertices(self) -> pd.DataFrame:
        table_name = f"{self.catalog}.{self.schema}.revit_room_vertices"
        sql = f"SELECT * FROM {table_name}"
        return self._execute_sql(sql)

    def get_room_edges(self) -> pd.DataFrame:
        table_name = f"{self.catalog}.{self.schema}.revit_room_edges"
        sql = f"SELECT * FROM {table_name}"
        return self._execute_sql(sql)