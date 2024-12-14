from phi.agent import Agent
from phi.knowledge.json import JSONKnowledgeBase
from phi.vectordb.pgvector import PgVector

knowledge_base = JSONKnowledgeBase(
    path="data/workout_data.json",
    # Table name: ai.json_documents
    vector_db=PgVector(
        table_name="json_documents",
        db_url="postgresql://postgres.ycdsjfvuvnyjcmvnhzrm:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres",
    ),
)

agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)
agent.knowledge.load(recreate=False)

agent.print_response("Ask me about something from the knowledge base")
