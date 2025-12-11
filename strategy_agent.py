import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the LLM (Gemini via LangChain)
# We use the 'flash' model because it is fast and efficient for this kind of analysis
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

def generate_business_strategy(summary_data, top_returns):
    """
    Analyzes financial data using LangChain + Gemini to give strategic advice.
    """
    try:
        # 1. Format the data for the AI
        # Convert dictionary to a clean string list
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in summary_data.items()])
        
        # Format top returns
        returns_str = "No major return issues."
        if top_returns and len(top_returns) > 0:
            returns_str = ""
            for item in top_returns[:3]:
                qty = item.get('quantity', item.get('return_quantity', 'N/A'))
                sku = item.get('sku', 'Unknown SKU')
                reason = item.get('return_category', item.get('reason', 'General Return'))
                returns_str += f"- SKU: {sku} | Qty: {qty} | Reason: {reason}\n"

        # 2. Define the LangChain Prompt Template
        prompt = PromptTemplate(
            input_variables=["metrics", "returns"],
            template="""
            You are a ruthless, high-level E-commerce CFO and Business Strategist.
            
            ### FINANCIAL SNAPSHOT
            {metrics}
            
            ### PROBLEM AREAS (High Returns)
            {returns}
            
            ### TASK
            Write a "Strategic Profit & Health Report" (in Markdown).
            
            1. **Financial Health Grade**: Give a grade (A, B, C, D, F) based on the Net Profit margin relative to Total Payment (Revenue). Explain why.
            2. **The "Leak" Detector**: Identify the single biggest expense draining profit right now (is it Ads, Returns, or COGS?).
            3. **Return Analysis**: Look at the top returned items. Suggest one specific operational fix (e.g., "Update sizing chart" or "Better packaging").
            4. **The Killer Move**: Give ONE aggressive, specific action to increase Net Profit by 10% next month.
            
            Keep it concise, professional, and actionable. Do not use fluff.
            """
        )
        
        # 3. Create the Chain
        # This connects the Prompt to the LLM
        chain = LLMChain(llm=llm, prompt=prompt)
        
        print("\n🧠 Strategy Agent is analyzing financials via LangChain...")
        
        # 4. Run the Chain
        # .invoke() is the modern way to run chains in LangChain
        result = chain.invoke({"metrics": metrics_str, "returns": returns_str})
        
        # 'result' is a dictionary containing the 'text' key
        return result['text']

    except Exception as e:
        print(f"Strategy Agent Error: {e}")
        return f"Error generating strategy: {str(e)}"