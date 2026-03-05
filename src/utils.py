import yaml
import json
import sqlite3
import pandas as pd

def load_config(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    db_path = config["data"]["database_path"]
    table = config["data"]["table_name"]
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def load_meta(config):
    with open(config["data"]["database_meta_path"], "r") as f:
        return json.load(f)
