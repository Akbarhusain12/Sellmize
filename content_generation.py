import requests
import json
from typing import Any, List, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_community.tools import DuckDuckGoSearchRun

# --- 1. Custom LangChain Wrapper for Pollinations (The "Engine") ---
# --- 1. Custom LangChain Wrapper for Pollinations (The "Engine") ---
class PollinationsLLM(LLM):
    """
    A custom LangChain wrapper for the free Pollinations.AI text API.
    """
    model_name: str = "openai"
    
    @property
    def _llm_type(self) -> str:
        return "pollinations"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the API call."""
        headers = {"Content-Type": "application/json"}
        
        # FIX: Removed 'temperature' to prevent 400 Bad Request Error
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "model": self.model_name,
            "jsonMode": False,
            "seed": 42
        }

        try:
            response = requests.post(
                "https://text.pollinations.ai/",
                json=payload,
                headers=headers,
                timeout=60 # Increased timeout for safety
            )
            response.raise_for_status() # This triggers the error if status is 400/500
            return response.text
        except Exception as e:
            # Return the error as string so the Agent catches it
            raise Exception(f"API Error: {str(e)}")

# --- 2. The Market Research Agent ---
class MarketAwareAgent:
    def __init__(self):
        self.llm = PollinationsLLM()
        self.search = DuckDuckGoSearchRun()
        print("✓ Agent Initialized with Market Search Capabilities")

    def analyze_market(self, product_name: str, category: str) -> str:
        """
        Searches Amazon listings to find winning keywords.
        """
        print(f"   ⟳ Searching market trends for: {product_name}...")
        
        # specific query to target Amazon listings
        query = f"site:amazon.in {product_name} {category} best selling product features description"
        
        try:
            # Get search snippets (Limit to first few results to save context window)
            search_results = self.search.invoke(query)
            return search_results
        except Exception as e:
            print(f"   ! Search failed: {e}")
            return "No market data available."

    def generate_listing(self, attributes: Dict[str, str]) -> Dict:
        """
        Main flow: Research -> Strategize -> Write (High Volume Mode)
        """
        product_name = attributes.get('Product Name', 'Generic Product')
        category = attributes.get('Category', 'General')
        
        # Step 1: Market Research
        competitor_data = self.analyze_market(product_name, category)
        
        # Step 2: Construct the "Mega Prompt"
        attr_string = "\n".join([f"- {k}: {v}" for k, v in attributes.items()])
        
        prompt = f"""
        Act as a Senior Amazon Copywriter.
        
        STEP 1: ANALYZE MARKET DATA
        "{competitor_data}"
        
        STEP 2: WRITE HIGH-CONVERSION LISTING
        Use the "User Product Details" below.
        
        CRITICAL BULLET POINT INSTRUCTIONS:
        1. Write exactly 5 bullet points.
        2. **LENGTH REQUIREMENT:** Each bullet point MUST be a full paragraph (3-4 sentences, approx 40-50 words).
        3. **STRUCTURE:** Start with an UPPERCASE HEADER. Follow with the Feature, then the Benefit, then a Real-World Use Case.
        4. Do NOT be concise. Be detailed, persuasive, and verbose.
        
        USER PRODUCT DETAILS:
        {attr_string}
        
        REQUIRED OUTPUT FORMAT (Valid JSON only):
        {{
            "market_analysis": "Strategy used...",
            "title": "SEO Title (150+ chars)...",
            "bullet_points": [
                "HEADER: First sentence explains the feature in depth. Second sentence explains why this matters to the user. Third sentence gives a specific example of use.",
                "HEADER: First sentence... Second sentence... Third sentence...",
                "HEADER: First sentence... Second sentence... Third sentence...",
                "HEADER: First sentence... Second sentence... Third sentence...",
                "HEADER: First sentence... Second sentence... Third sentence..."
            ],
            "description": "3-4 rich paragraphs..."
        }}
        """

        # Step 3: Generate
        print("   ⟳ Synthesizing listing with market data...")
        response_text = self.llm.invoke(prompt)
        
        # Step 4: Robust JSON Parsing
        try:
            # Clean Markdown wrappers
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            
            import re
            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return json.loads(clean_text)
                
        except json.JSONDecodeError:
            return {
                "market_analysis": "AI generation successful, but formatting failed.",
                "title": f"Listing for {product_name}",
                "bullet_points": ["Error parsing bullet points - please try again."],
                "description": response_text,
                "error": "JSON_PARSE_ERROR"
            }


def generate_product_content(attributes: dict) -> dict:
    """
    Wrapper function that allows application.py to use the 
    new MarketAwareAgent without changing any code.
    """
    try:
        # 1. Initialize the Agent
        agent = MarketAwareAgent()
        
        # 2. Run the Generation
        # We pass the attributes directly. The agent uses keys like 'Product Name'.
        # If your frontend sends different keys, they are passed along here.
        result_data = agent.generate_listing(attributes)
        
        # 3. Check for internal errors from the agent
        if "error" in result_data:
            return {
                "success": False,
                "error": result_data["error"]
            }
            
        # 4. Return success format expected by application.py
        return {
            "success": True,
            "content": result_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Agent Error: {str(e)}"
        }
        