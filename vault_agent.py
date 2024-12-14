from phi.agent import Agent
from phi.model.openai import OpenAIChat

from vault_tools import VaultAPITools

# Create agent with specific tools enabled
vault_tools = VaultAPITools(
    enable_entries=True,
    enable_history=True
)

agent = Agent(
    name="Vault Analyst",
    model=OpenAIChat(model="gpt-4"),
    tools=[vault_tools],
    instructions=[
        "You analyze data from Vault API",
        "Always verify data availability before analysis",
        "Provide clear error messages when needed"
    ],
    show_tool_calls=True
)