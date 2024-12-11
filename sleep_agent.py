import json
import logging
from typing import Any, Dict, List

import dotenv
from phi.agent import Agent
from phi.knowledge.json import JSONKnowledgeBase
from phi.model.openai import OpenAIChat
from phi.vectordb.pgvector import PgVector, SearchType

dotenv.load_dotenv(".env")


def process_sleep_data() -> List[Dict[str, Any]]:
    """
    Preprocess sleep data to create smaller, focused documents
    
    Returns:
        List[Dict[str, Any]]: List of processed sleep records
    """
    logging.info("Starting data preprocessing...")
    with open("data/sleep_data.json", "r") as f:
        data = json.load(f)
    
    documents = []
    # Process current entry
    if "current" in data:
        logging.info("Processing current entry")
        documents.append(data["current"])
    
    # Process historical entries
    if "history" in data:
        logging.info(f"Processing {len(data['history'])} historical entries")
        documents.extend(data["history"])
    
    logging.info(f"Processed total of {len(documents)} documents")
    return documents


def create_knowledge_base() -> JSONKnowledgeBase:
    """
    Creates a knowledge base with preprocessed data
    
    Returns:
        JSONKnowledgeBase: Configured knowledge base instance
    """
    processed_data = process_sleep_data()
    
    # Save processed data
    with open("data/processed_sleep_data.json", "w") as f:
        json.dump(processed_data, f)
    
    return JSONKnowledgeBase(
        path="data/processed_sleep_data.json",
        vector_db=PgVector(
            table_name="sleep_data",
            schema="ai",
            db_url="postgresql://postgres:FHA8HuCkgYbwOaiqsMb3Z7SOl90Bh1QL@junction.proxy.rlwy.net:44594/railway",
            search_type=SearchType.hybrid,
            vector_score_weight=0.5
        ),
    )


# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize knowledge base and agent
knowledge_base = create_knowledge_base()
agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)

# Load data with recreation
agent.knowledge.load(recreate=True)

agent.print_response("How is my sleep quality?")
