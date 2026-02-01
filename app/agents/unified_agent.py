import os
import logging
import json
import asyncio
import re
from typing import Dict, Any, List

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SellMizeAssistant:
    """
    Unified Intelligence Engine for SellMize.
    Combines Market Analysis, Product Scouting, and Content Generation.
    """

    def __init__(self):
        # 1. Initialize Core LLM (Fast & Smart) - Replaces Pollinations
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # 2. Initialize Search Tool (Tavily is better for data than DDG)
        self.search_tool = TavilySearchResults(
            max_results=7,
            include_answer=True,
            include_raw_content=True
        )

    async def process_request(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Main Entry Point: Routes the request to the correct logic.
        """
        try:
            intent = mode.upper()
            
            # Step A: Auto-Classify if needed
            if intent == "AUTO":
                intent = await self._classify_intent(query)
                logger.info(f"🧠 Intent Classified: {intent}")

            # Step B: Route Logic
            if intent == "CONTENT":
                return await self._run_content_gen(query)
            elif intent == "PRODUCT":
                return await self._run_product_scout(query)
            elif intent == "RESEARCH":
                return await self._run_market_analysis(query)
            elif intent == "STRATEGY":
                return await self._run_strategy(query)
            else:
                return await self._run_chat(query)

        except Exception as e:
            logger.error(f"Critical Error in SellMizeAssistant: {e}", exc_info=True)
            return {"type": "error", "content": f"An error occurred: {str(e)}"}

    # =========================================================================
    # 🧠 LOGIC 1: INTENT CLASSIFIER
    # =========================================================================
    async def _classify_intent(self, query: str) -> str:
        prompt = f"""
        Classify this E-commerce user query into one of these categories:
        
        1. CONTENT: Writing Amazon listings, titles, bullet points, emails. ("Write a listing for...", "Generate description")
        2. PRODUCT: Finding specific items, prices, specs, comparisons. ("Best gaming laptop under $1000", "Price of iPhone")
        3. RESEARCH: Market trends, competitors, niche analysis. ("Is coffee dropshipping dead?", "Competitors for XYZ")
        4. STRATEGY: Business math, ROI, margins, general advice. ("How to calculate profit?", "FBA vs FBM")
        5. CHAT: General greetings or questions.

        Query: "{query}"

        Return ONLY the category name. No punctuation.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip().upper()

    # =========================================================================
    # 📝 LOGIC 2: CONTENT GENERATOR (High-Quality Listing)
    # =========================================================================
    async def _run_content_gen(self, query: str) -> Dict[str, Any]:
        """
        Generates structured JSON for Amazon Listings.
        """
        logger.info(f"📝 Generating Content for: {query}")
        
        # We assume the query contains the product details. 
        # In a real app, you might search for keywords first, but here we generate directly.
        
        prompt = f"""
        Act as a Senior Amazon Copywriter.
        User Request/Product Details: "{query}"
        
        Generate a high-conversion Amazon listing.
        
        CRITICAL INSTRUCTIONS:
        1. Write exactly 5 bullet points.
        2. **LENGTH:** Each bullet point must be substantial (2-3 sentences).
        3. **STRUCTURE:** Start bullets with an UPPERCASE HEADER.
        4. **KEYWORDS:** Include high-volume keywords naturally.

        REQUIRED OUTPUT FORMAT (Valid JSON Only):
        {{
            "title": "SEO Optimized Title (150+ chars)...",
            "bullet_points": [
                "HEADER: First sentence... Second sentence...",
                "HEADER: First sentence... Second sentence...",
                "HEADER: First sentence... Second sentence...",
                "HEADER: First sentence... Second sentence...",
                "HEADER: First sentence... Second sentence..."
            ],
            "description": "3-4 rich paragraphs in HTML format (using <br> and <b> tags).",
            "keywords": "comma, separated, list, of, keywords"
        }}
        """
        
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        json_data = self._clean_and_parse_json(res.content)
        
        if json_data:
            return {"type": "content", "content": json_data}
        else:
            # Fallback if JSON fails
            return {"type": "chat", "content": res.content}

    # =========================================================================
    # 🕵️ LOGIC 3: PRODUCT SCOUT (Finds Items)
    # =========================================================================
    async def _run_product_scout(self, query: str) -> Dict[str, Any]:
        logger.info(f"🚀 Scouting Products for: {query}")
        
        # 1. Search
        search_query = f"{query} price specifications reviews buy online site:amazon.in OR site:flipkart.com"
        raw_results = await asyncio.to_thread(self.search_tool.invoke, search_query)
        
        if not raw_results:
            return {"type": "chat", "content": "I couldn't find specific products. Try being more specific."}

        # 2. Extract Data to JSON
        context = str(raw_results)[:4000]
        prompt = f"""
        Extract top products from these search results: {context}
        Query: "{query}"
        
        Return a valid JSON List of objects:
        [
            {{
                "name": "Specific Model Name",
                "price": "Price (number only)",
                "currency": "INR/USD",
                "rating": "4.5",
                "features": ["Feature 1", "Feature 2"],
                "link": "Link from context if available, else null"
            }}
        ]
        JSON ONLY.
        """
        
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        products = self._clean_and_parse_json(res.content)
        
        # 3. Generate a summary text
        summary_prompt = f"Summarize these products for the user briefly: {json.dumps(products)}"
        summary = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])

        return {
            "type": "product",
            "content": summary.content, # Text summary
            "products": products or []  # Structured data for UI cards
        }

    # =========================================================================
    # 📈 LOGIC 4: MARKET RESEARCH (Deep Analysis)
    # =========================================================================
    async def _run_market_analysis(self, query: str) -> Dict[str, Any]:
        logger.info(f"📊 Analyzing Market: {query}")
        
        # 1. Broad Search
        search_query = f"{query} market size trends competitors statistics 2025"
        raw_results = await asyncio.to_thread(self.search_tool.invoke, search_query)
        
        # 2. Generate Report
        context = str(raw_results)[:5000]
        prompt = f"""
        You are 'MarketLens', a Senior Business Analyst.
        Topic: "{query}"
        Search Data: {context}
        
        Write a strategic report in Markdown.
        Structure:
        1. **Executive Summary**
        2. **Market Dynamics** (Trends)
        3. **Competitor Analysis**
        4. **Strategic Recommendation**
        
        Professional tone. Use bolding and lists.
        """
        
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return {"type": "report", "content": res.content}

    # =========================================================================
    # 💬 LOGIC 5: STRATEGY & CHAT
    # =========================================================================
    async def _run_strategy(self, query: str) -> Dict[str, Any]:
        prompt = f"Act as an MBA E-commerce Coach. Give concise, bullet-point advice for: {query}"
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return {"type": "chat", "content": res.content}

    async def _run_chat(self, query: str) -> Dict[str, Any]:
        prompt = f"You are SellMize AI. Be helpful and concise. User: {query}"
        res = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return {"type": "chat", "content": res.content}

    # =========================================================================
    # 🛠️ HELPER: JSON PARSER
    # =========================================================================
    def _clean_and_parse_json(self, text: str) -> Any:
        """
        Robustly extracts JSON from LLM response using Regex.
        Handles cases where LLM adds Markdown ```json ... ``` or extra text.
        """
        try:
            # 1. Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            # 2. Regex search for { ... } or [ ... ]
            json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
            
        try:
            # 3. Cleanup Markdown code blocks
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except:
            logger.error(f"Failed to parse JSON from: {text[:100]}...")
            return None

# Simple usage example for testing
if __name__ == "__main__":
    agent = SellMizeAssistant()
    # async run wrapper
    # result = asyncio.run(agent.process_request("Best gaming laptop under 80000", mode="PRODUCT"))
    # print(result)