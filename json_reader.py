import json
import logging
import os
from typing import Iterator, List

import httpx
from dotenv import load_dotenv
from phi.agent import Agent
from phi.document import Document
from phi.document.reader import Reader
from phi.knowledge import AgentKnowledge
from phi.knowledge.json import JSONKnowledgeBase
from phi.vectordb.pgvector import PgVector

load_dotenv(".env")

# Reader can be paginated!!
class APIReader(Reader):
    access_token: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__client = httpx.Client(
            base_url="http://localhost:3001",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        self.__iteration_started = False
        self.__pagination_token = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__client.__exit__(exc_type, exc_val, exc_tb)

    def can_iterate(self):
        return not self.__iteration_started or self.__pagination_token is not None

    def read(self, path: str) -> List[Document]:
        params = {
            "limit": 100,
        }

        if self.__pagination_token is not None:
            params["next_token"] = self.__pagination_token
        self.__iteration_started = True

        response = self.__client.get(path, params=params).json()
        res = []
        for item in response["data"]:
            res.append(Document(
                # 2 strategies - deterministic lookup ID, or autogenerated ID. It depends on what we're looking for.
                id=item["lookup_id"],
                content=json.dumps(item),
                meta_data={"type": "historical"},
                name="sleep",
                # What does embedding, embedder, usage, reranking_score do?
            ))
        self.__pagination_token = response["pagination"]["next"]
        return res


class APIKnowledgeBase(AgentKnowledge):
    path: str
    reader: APIReader

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        # Should raise StopIteration to end the loop
        while self.reader.can_iterate():
            yield self.reader.read(self.path)


def create_knowledge_base() -> APIKnowledgeBase:
    return APIKnowledgeBase(
        path="/entries/by_key/sleep",
        vector_db=PgVector(
            table_name="sleep_data",
            db_url="postgresql://anlyst:dev-postgresql@localhost:3007/global"
        ),
        reader=APIReader(
            access_token=os.getenv("ACCESS_TOKEN"),
        ),
    )

knowledge_base = create_knowledge_base()
agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)

# First run only
agent.knowledge.load(upsert=True)
# agent.print_response("How much did I sleep in the past 7 days?")