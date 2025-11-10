import os
import json
import re
from google import generativeai as genai
from config import Config


class ProductContentGenerator:
    """
    AI-powered product content generator using Google Gemini API.
    """
    
    def __init__(self):
        """Initialize the content generator with Gemini API."""
        # Check if API key is set
        if not hasattr(Config, 'GEMINI_API_KEY') or not Config.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY not set in Config.")
            self.available = False
        else:
            self.available = True
            try:
                # Configure the API with key from Config
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                print("✓ Gemini API initialized successfully")
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.available = False
    
    def generate_content(self, product_attributes: dict) -> dict:
        """
        Generate product title, description, and bullet points.
        
        Args:
            product_attributes: Dictionary with product details
            
        Returns:
            Dictionary with generated content
        """
        if not self.available:
            return {
                'success': False,
                'error': 'Content generator is not available. Please check GEMINI_API_KEY in config.'
            }
        
        try:
            # Filter out empty values
            filtered_attrs = {k: v for k, v in product_attributes.items() if v and str(v).strip()}
            
            if not filtered_attrs:
                return {
                    'success': False,
                    'error': 'No product attributes provided'
                }
            
            # Format product details
            product_details = "\n".join([f"{k}: {v}" for k, v in filtered_attrs.items()])
            
            # Build the prompt
            prompt = f"""You are a professional AI copywriter who writes e-commerce product content.

Generate creative, persuasive, and SEO-friendly product content strictly in JSON format.

### Instructions:
1. Write a short and catchy product title (with brand name and main feature).
2. Write a two-paragraph product description:
   - Paragraph 1: Introduce the product, highlight its emotional and lifestyle appeal.
   - Paragraph 2: Explain its features, benefits, and what makes it stand out.
3. Write 5 longer bullet points (1–2 sentences each) describing key highlights and unique selling points.

### Product Details:
{product_details}

### Output JSON Example:
{{
  "title": "Generated product title",
  "description": "Two engaging paragraphs here...",
  "bullet_points": [
    "Detailed bullet point 1.",
    "Detailed bullet point 2.",
    "Detailed bullet point 3.",
    "Detailed bullet point 4.",
    "Detailed bullet point 5."
  ]
}}

Return only valid JSON — no extra text, markdown, or explanations."""

            # Generate content
            response = self.model.generate_content(prompt)
            
            # Extract text
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*$', '', text)
            text = text.strip()
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return {
                        'success': True,
                        'content': {
                            'title': data.get('title', ''),
                            'description': data.get('description', ''),
                            'bullet_points': data.get('bullet_points', [])
                        }
                    }
                except json.JSONDecodeError as e:
                    return {
                        'success': False,
                        'error': f'Failed to parse JSON response: {str(e)}',
                        'raw_output': text
                    }
            else:
                return {
                    'success': False,
                    'error': 'No JSON found in model output',
                    'raw_output': text
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Error generating content: {str(e)}'
            }


# Helper function for Flask integration
def generate_product_content(product_attributes: dict) -> dict:
    """
    Generate product content.
    
    Args:
        product_attributes: Dictionary with product details
        
    Returns:
        Dictionary with generated content or error
    """
    generator = ProductContentGenerator()
    return generator.generate_content(product_attributes)