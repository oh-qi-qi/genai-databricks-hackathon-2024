import requests
import pandas as pd
import json
import sys
import os
from typing import List, Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, LongType
from pyspark.sql.functions import from_json, col

class DataRetrievalTool:
    def __init__(self, databricks_url: str, token: str, warehouse_id: str, catalog: str, schema: str, spark: SparkSession):
        self.databricks_url = databricks_url.rstrip('/')
        self.token = token
        self.warehouse_id = warehouse_id
        self.catalog = catalog
        self.schema = schema
        self.spark = spark
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

    def _create_spark_table(self, df: pd.DataFrame, table_name: str) -> None:
        # Initially, create all columns as StringType
        schema = StructType([StructField(col, StringType(), True) for col in df.columns])

        # Convert pandas DataFrame to Spark DataFrame
        spark_df = self.spark.createDataFrame(df.astype(str), schema)

        # Define the bounds schema
        bounds_schema = ArrayType(StructType([
            StructField("X", DoubleType(), True),
            StructField("Y", DoubleType(), True),
            StructField("Z", DoubleType(), True)
        ]))

        # Cast columns to appropriate types
        for column in spark_df.columns:
            if column in ['id', 'src', 'dst', 'door_id']:
                spark_df = spark_df.withColumn(column, col(column).cast(LongType()))
            elif column in ['bounds', 'src_bounds', 'dst_bounds', 'door_bounds']:
                spark_df = spark_df.withColumn(column, from_json(col(column), bounds_schema))

        # Create a temporary view
        spark_df.createOrReplaceTempView(table_name)

        print(f"Created temporary view: {table_name}")
        print(f"Schema of {table_name}:")
        spark_df.printSchema()

    def get_room_vertices(self) -> None:
        table_name = f"{self.catalog}.{self.schema}.revit_room_vertices"
        sql = f"SELECT * FROM {table_name}"
        df = self._execute_sql(sql)
        self._create_spark_table(df, "room_vertices")

    def get_room_edges(self) -> None:
        table_name = f"{self.catalog}.{self.schema}.revit_room_edges"
        sql = f"SELECT * FROM {table_name}"
        df = self._execute_sql(sql)
        self._create_spark_table(df, "room_edges")
        
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, main_dir)

# Usage example
from common.databricks_config import (
    DATABRICKS_URL, 
    TOKEN, 
    DATABRICKS_WAREHOUSE_ID,
    catalog_name, 
    schema_name
)

# Create a SparkSession
spark = SparkSession.builder.appName("RoomDataRetrieval").getOrCreate()

data_retrieval_tool = DataRetrievalTool(DATABRICKS_URL, TOKEN, DATABRICKS_WAREHOUSE_ID, catalog_name, schema_name, spark)

try:
    print("Retrieving vertices data...")
    data_retrieval_tool.get_room_vertices()
    
    print("\nRetrieving edges data...")
    data_retrieval_tool.get_room_edges()
    
    print("\nData retrieval complete. Spark tables created.")
    
    # Example of using the created Spark tables
    print("\nSample data from room_vertices:")
    spark.sql("SELECT * FROM room_vertices LIMIT 5").show(truncate=False)
    
    print("\nSample data from room_edges:")
    spark.sql("SELECT * FROM room_edges LIMIT 5").show(truncate=False)

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    # Don't forget to stop the SparkSession when you're done
    spark.stop()