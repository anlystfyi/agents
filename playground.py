import os

from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.playground import Playground, serve_playground_app
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.slack import SlackTools

load_dotenv()

slack_token = os.getenv("SLACK_TOKEN")
if not slack_token:
    raise ValueError("SLACK_TOKEN not set")

slack_agent = Agent(
    name="Slack Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[SlackTools(slack_token)],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[slack_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
