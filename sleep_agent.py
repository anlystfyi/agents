from typing import Any, Dict

from phi.agent import Agent
from phi.knowledge.json import JSONKnowledgeBase
from phi.model.openai import OpenAIChat
from phi.vectordb.pgvector import PgVector


def create_knowledge_base() -> JSONKnowledgeBase:
    """
    Creates a knowledge base with proper ID mapping
    """
    return JSONKnowledgeBase(
        path="data/sleep_data.json",
        vector_db=PgVector(
            table_name="sleep_data",
            db_url="postgresql://postgres:FHA8HuCkgYbwOaiqsMb3Z7SOl90Bh1QL@junction.proxy.rlwy.net:44594/railway"
        ),
    )

knowledge_base = create_knowledge_base()
agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)

# First run only
agent.knowledge.load(recreate=True)

agent.print_response("Ask me about something from the knowledge base")