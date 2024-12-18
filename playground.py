from typing import List

from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.playground import Playground, serve_playground_app
from phi.storage.agent.sqlite import SqlAgentStorage

from tools.sleep_tools import SleepAPITools

load_dotenv()

def create_sleep_analyst() -> Agent:
    """Create a sleep analysis agent with persistent storage"""
    return Agent(
        name="Sleep Analyst",
        model=OpenAIChat(model="gpt-4"),
        tools=[SleepAPITools(use_vector_db=False)],
        storage=SqlAgentStorage(
            table_name="sleep_analyst", 
            db_file="agents.db"
        ),
        instructions=[
            "You are a sleep data analyst specializing in analyzing sleep patterns from Vault API data.",
            "Available commands:",
            "- get_sleep_data(days): Fetch raw sleep data for specified days",
            "- get_sleep_analysis(): Analyze most recent sleep entry",
            "- get_recent_sleep_trends(days): Analyze sleep trends over time",
            "- search_sleep_by_date(start_date, end_date, order='desc', limit=100, max_pages=5): Search sleep entries by date range",
            "",
            "Search Examples:",
            '1. Recent entries: search_sleep_by_date(order="desc")',
            '2. Date range: search_sleep_by_date("2023-01-01", "2023-12-31")',
            '3. Paginated search: search_sleep_by_date("2024-05-01", "2024-05-31", limit=100, max_pages=5)',
            "",
            "When analyzing sleep data:",
            "1. Always verify data availability before analysis",
            "2. Present insights in markdown tables and bullet points",
            "3. Include timestamps for data freshness",
            "4. Focus on actionable metrics like:",
            "   - Sleep duration trends",
            "   - Sleep quality patterns",
            "   - Consistency of sleep schedule",
            "5. Provide specific recommendations based on the data",
            "6. For large date ranges, mention if there might be more data available",
            "",
            "When presenting sleep data:",
            "1. Use the formatted table from the 'display' field in the response",
            "2. Add your analysis below the table",
            "3. Highlight any notable patterns or outliers",
            "4. Include the summary statistics in your analysis",
            "5. If data spans multiple pages, mention the total available entries",
            "",
            "Example Response Format:",
            "Here's your sleep data for [date range]:",
            "[Insert formatted table]",
            "",
            "Analysis:",
            "- Key observations about sleep patterns",
            "- Notable trends in quality and duration",
            "- Recommendations based on the data"
        ],
        add_history_to_messages=True,
        markdown=True,
        show_tool_calls=True,
    )

def get_available_agents() -> List[Agent]:
    """Get list of available agents for the playground"""
    return [create_sleep_analyst()]

# Add test function
def test_sleep_api():
    """Test the sleep API connection"""
    tools = SleepAPITools()
    print("Testing sleep API connection...")
    result = tools.get_sleep_data(days=1)
    print(f"API Response: {result}")

# Create the playground app
app = Playground(
    agents=get_available_agents(),
).get_app()

if __name__ == "__main__":
    # Add test before starting server
    test_sleep_api()
    serve_playground_app(
        "playground:app",
        reload=True,
        port=7777
    )
