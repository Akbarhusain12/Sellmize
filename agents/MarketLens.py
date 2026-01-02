import os
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load Env
load_dotenv()

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedMarketAgent:
    """
    A unified agent that handles:
    1. Market Research (Trends, Competitors, Risks)
    2. Product Scouting (Specs, Prices, Comparisons)
    3. General Conversation
    """

    def __init__(self):
        # 1. Initialize Core LLM (Fast & Smart)
        self.llm = ChatGroq(
            temperature=0.1, # Keep it factual
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # 2. Initialize Search Tool
        self.search_tool = TavilySearchResults(
            max_results=7,
            include_answer=True,
            include_raw_content=True
        )

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Main Entry Point: Classifies intent and routes to the correct logic.
        """
        try:
            # Step A: Classify Intent
            intent = await self._classify_intent(user_input)
            logger.info(f"🧠 Intent Classified: {intent}")

            # Step B: Route Logic
            if intent == "PRODUCT_SCOUT":
                return await self._run_product_scout(user_input)
            elif intent == "MARKET_ANALYSIS":
                return await self._run_market_analysis(user_input)
            else:
                return await self._run_chat(user_input)

        except Exception as e:
            logger.error(f"Critical Error in UnifiedAgent: {e}", exc_info=True)
            return {"type": "error", "content": f"An error occurred: {str(e)}"}

    # =========================================================================
    # 🧠 LOGIC 1: INTENT CLASSIFIER
    # =========================================================================
    async def _classify_intent(self, query: str) -> str:
        prompt = f"""
        You are a Router AI. Classify the user query into one of these 3 categories:

        1. PRODUCT_SCOUT: The user is looking for specific physical products, prices, specs, comparisons, or "best X under $Y".
           Examples: "Best gaming laptop under 50k", "iPhone 15 vs S23", "Price of Sony headphones".

        2. MARKET_ANALYSIS: The user asks about business ideas, industry trends, niche viability, dropshipping, or risks.
           Examples: "Is dropshipping dead?", "Trends in coffee industry", "Competitors for AI fitness app".

        3. CHAT: Greetings, compliments, or questions about you.
           Examples: "Hi", "Who are you?", "Thanks".

        Query: "{query}"

        Return ONLY the category name. No punctuation.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip().upper()

    # =========================================================================
    # 🕵️ LOGIC 2: PRODUCT SCOUT (The "Shopping" Mode)
    # =========================================================================
    async def _run_product_scout(self, query: str) -> Dict[str, Any]:
        logger.info("🚀 Running Product Scout Mode")
        
        # 1. Smart Search Query
        search_keywords = f"{query} price specifications reviews buy online site:amazon.in OR site:flipkart.com OR site:reliancedigital.in"
        raw_results = await asyncio.to_thread(self.search_tool.invoke, search_keywords)
        
        if not raw_results:
            return {"type": "product", "content": "I couldn't find any products matching that specific query.", "products": []}

        # 2. Extract Data (JSON)
        products = await self._extract_products_json(raw_results, query)
        
        # 3. Generate Report (Markdown)
        report = await self._generate_product_report(query, products)
        
        return {
            "type": "product",
            "content": report,
            "products": products
        }

    async def _extract_products_json(self, search_results: List[Dict], query: str) -> List[Dict]:
        context = str(search_results)[:4500] # Context limit
        
        prompt = f"""
        Extract top products from search results for query: "{query}".
        Context: {context}
        
        Return a valid JSON List of objects with:
        - name (Specific model)
        - price (Number only, ignore currency symbols)
        - currency (e.g. INR, USD)
        - rating (0-5 number)
        - features (List of 3 key specs)
        - brand
        
        Clean data only. If price is missing, skip or estimate.
        JSON ONLY.
        """
        try:
            res = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Basic cleaning
            json_str = res.content.strip()
            if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str)
        except:
            return []

    async def _generate_product_report(self, query: str, products: List[Dict]) -> str:
        product_context = json.dumps(products, indent=2)
        prompt = f"""
        Act as a Shopping Expert. 
        User Query: "{query}"
        Product Data: {product_context}
        
        Write a Buyer's Guide in Markdown.
        Structure:
        ## Executive Summary
        ## Feature Comparison Table (Use Markdown Table)
        ## Value for Money Analysis
        ## Final Verdict
        
        No conversational filler.
        """
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return res.content

    # =========================================================================
    # 📈 LOGIC 3: MARKET LENS (The "Strategy" Mode)
    # =========================================================================
    async def _run_market_analysis(self, query: str) -> Dict[str, Any]:
        logger.info("🚀 Running Market Analysis Mode")
        
        # 1. Broad Search
        search_query = f"{query} market size trends competitors pain points statistics"
        raw_results = await asyncio.to_thread(self.search_tool.invoke, search_query)
        
        # 2. Generate Deep Report
        context = str(raw_results)[:5000]
        
        prompt = f"""
        You are 'MarketLens', a Senior Business Analyst.
        Topic: "{query}"
        Search Data: {context}
        
        Write a strategic report in Markdown.
        
        Guidelines:
        1. **Executive Summary**: 3 sentences on the opportunity.
        2. **Market Dynamics**: Trends and Shift.
        3. **Competitor Analysis**: Who dominates?
        4. **Visuals**: If a diagram helps explain a concept (like a growth cycle or supply chain), insert a tag like .
        5. **Strategic "Wedge"**: How to enter this market?
        
        Professional tone. No fluff.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "type": "market",
            "content": response.content,
            "products": [] # No specific products for market analysis
        }

    # =========================================================================
    # 💬 LOGIC 4: CHAT (The "Conversational" Mode)
    # =========================================================================
    async def _run_chat(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        You are SellMize AI, a helpful E-commerce Assistant.
        User said: "{query}"
        
        Reply helpfully and briefly. If they ask about products or business, ask them to be specific.
        """
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "type": "chat",
            "content": res.content,
            "products": []
        }