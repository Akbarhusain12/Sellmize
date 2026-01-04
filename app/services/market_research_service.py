from app.agents.marketlens_agent import UnifiedMarketAgent


async def handle_marketlens_query(query: str):
    agent = UnifiedMarketAgent()
    return await agent.process_request(query)
