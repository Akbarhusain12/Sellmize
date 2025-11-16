import os
import json
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 1. Define the desired JSON output structure
class ProductContent(BaseModel):
    title: str = Field(description="Generated product title")
    description: str = Field(description="Two engaging paragraphs here...")
    bullet_points: List[str] = Field(description="List of 5 detailed bullet points")

class ProductContentGenerator:
    """
    AI-powered product content generator using LangChain and Google Gemini.
    """
    
    def __init__(self):
        """Initialize the content generator with LangChain."""
        if not hasattr(Config, 'GEMINI_API_KEY') or not Config.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not set in Config.")
            self.available = False
            self.chain = None
        else:
            self.available = True
            try:
                # 2. Initialize the Model
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    api_key=Config.GEMINI_API_KEY,
                    temperature=0.2
                )
                
                # 3. Initialize the Output Parser
                self.parser = JsonOutputParser(pydantic_object=ProductContent)
                
                # 4. Create the Prompt Template
                prompt = PromptTemplate(
                    template="""You are a professional AI copywriter who writes e-commerce product content.
Generate creative, persuasive, and SEO-friendly product content strictly in JSON format.

### Instructions:
1. Write a short and catchy product title.
2. Write a two-paragraph product description (lifestyle + features).
3. Write 5 detailed bullet points (1-2 sentences each).

### Product Details:
{product_details}

### Output Format:
{format_instructions}
""",
                    input_variables=["product_details"],
                    # 5. Connect the parser's instructions to the prompt
                    partial_variables={"format_instructions": self.parser.get_format_instructions()},
                )
                
                # 6. Create the Chain
                # This "pipe" connects all the steps together
                self.chain = prompt | llm | self.parser
                
                print("✓ LangChain (Gemini) initialized successfully")

            except Exception as e:
                print(f"Error initializing LangChain: {e}")
                self.available = False

    def generate_content(self, product_attributes: dict) -> dict:
        """
        Generate product title, description, and bullet points.
        """
        if not self.available or not self.chain:
            return {
                'success': False,
                'error': 'Content generator is not available. Please check GEMINI_API_KEY in config.'
            }
        
        try:
            filtered_attrs = {k: v for k, v in product_attributes.items() if v and str(v).strip()}
            if not filtered_attrs:
                return {'success': False, 'error': 'No product attributes provided'}
            
            product_details = "\n".join([f"{k}: {v}" for k, v in filtered_attrs.items()])
            
            # 7. Run the chain!
            # LangChain automatically runs the prompt, calls the model,
            # and parses the JSON output in one step.
            generated_data = self.chain.invoke({"product_details": product_details})
            
            return {
                'success': True,
                'content': generated_data  # This is already a clean Python dictionary
            }
                
        except Exception as e:
            # This can catch errors from the model or from the parser
            return {
                'success': False,
                'error': f'Error generating content: {str(e)}'
            }

# Helper function for Flask integration
def generate_product_content(product_attributes: dict) -> dict:
    generator = ProductContentGenerator()
    return generator.generate_content(product_attributes)