import os

from phi.agent import Agent
from phi.tools.slack import SlackTools

slack_token = os.getenv("SLACK_TOKEN")
if not slack_token:
    raise ValueError("SLACK_TOKEN not set")
slack_tools = SlackTools(slack_token)



# Example 1: Send a message to a Slack channel
agent.print_response("Send a message 'Hello from Anlyst Agent!' to the channel #all-anlyst", markdown=True)

# Example 2: List all channels in the Slack workspace
agent.print_response("List all channels in our Slack workspace", markdown=True)

# Example 3: Get the message history of a specific channel
agent.print_response("Get the last 10 messages from the channel #brainstorming", markdown=True)

