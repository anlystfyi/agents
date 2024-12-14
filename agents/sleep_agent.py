"""Sleep data analysis agent using REST API and DuckDB."""

import json
import os
from typing import Any, Dict, List, Optional

import duckdb
import requests
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckdb import DuckDbTools


def get_sleep_data() -> List[Dict[str, Any]]:
    """Fetch sleep data from the Vault API."""
    token = os.getenv("VAULT_API_KEY")
    if not token:
        raise ValueError("VAULT_API_KEY environment variable not set")
    
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(
            "https://vault-api.anlyst.ai/entries/by_key/sleep",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()["data"]
        print(f"Successfully fetched {len(data)} sleep records")
        return data
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise

def setup_duckdb() -> DuckDbTools:
    """Setup DuckDB with sleep data."""
    try:
        # Get the data first
        sleep_data = get_sleep_data()
        
        # Create a connection and verify data
        conn = duckdb.connect(":memory:")
        
        # Create DuckDB tools
        tools = DuckDbTools(
            connection=conn,
            tables={
                "raw_sleep": sleep_data
            }
        )
        
        # Transform data
        tools.run_sql("""
            CREATE OR REPLACE TABLE sleep_metrics AS 
            SELECT 
                CAST(start_time AS TIMESTAMP) as start_time,
                CAST(end_time AS TIMESTAMP) as end_time,
                CAST(duration AS DOUBLE) / 3600 as duration_hours,
                CAST(quality AS INTEGER) as quality,
                CAST(respiratory_rate AS DOUBLE) as respiratory_rate,
                CAST(heart_rate_avg AS DOUBLE) as heart_rate_avg,
                CAST(heart_rate_max AS DOUBLE) as heart_rate_max
            FROM raw_sleep;
        """)
        
        return tools
    except Exception as e:
        print(f"Error setting up DuckDB: {str(e)}")
        raise

def create_sleep_agent() -> Agent:
    """Create an agent for sleep data analysis."""
    try:
        # Setup DuckDB tools
        duckdb_tools = setup_duckdb()
        
        return Agent(
            name="Sleep Analyst",
            role="Analyze sleep patterns and provide insights",
            model=OpenAIChat(model="gpt-4"),
            tools=[duckdb_tools],
            instructions=[
                "You are a sleep analysis expert.",
                "Analyze sleep data from the sleep_metrics table.",
                "Use SQL to query the data and create visualizations.",
                "Duration is in hours.",
                "Quality is on a scale of 0-100.",
                "Always explain your methodology."
            ],
            markdown=True,
            show_tool_calls=True,
        )
    except Exception as e:
        print(f"Error creating agent: {str(e)}")
        raise

def main():
    """Main function to run the sleep agent."""
    try:
        agent = create_sleep_agent()
        agent.print_response(
            "What is the average sleep duration in the dataset? "
            "Show me a simple summary.",
            stream=True,
        )
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()