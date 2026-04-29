import ollama
import json
import re

class Tier2LLM:
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.client = ollama.Client(host=url)

    def generate_prompt(self, text: str, categories: list):
        prompt = f"""
Task: Classify the following text into one or more of the specified categories. provide CoT reasoning.
Available Categories: {', '.join(categories)}

Input Text: "{text}"

Output Format (STRICT JSON):
{{
  "labels": ["Category1", "Category2"],
  "confidence": 0.95,
  "reasoning": "Explain why this label was chosen based on the text.",
  "new_category_suggestion": "Optional category name if none of the above fit perfectly"
}}
"""
        return prompt

    def predict(self, text: str, categories: list):
        prompt = self.generate_prompt(text, categories)
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt)
            response_text = response['response']
            
            # Extract JSON from response (handling potential markdown formatting)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result, response_text
            else:
                return None, response_text
        except Exception as e:
            print(f"Error in Tier 2 LLM: {e}")
            return None, str(e)
