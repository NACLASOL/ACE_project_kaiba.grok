import os
import json
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook
from parse_data import rubric_bullets


load_dotenv()


class StrippingLiteLLMClient(LiteLLMClient):
    '''Custom LiteLLMClient that strips Markdown fences from LLM response before returning'''
    def query(self, prompt, **kwargs):
        response = super().query(prompt, **kwargs)

        # Strips common Markdown fences ('''...''').
        if isinstance(response, dict) and 'choices' in response:
            text = response['choices'][0]['message']['content']
        else:
            text = str(response)
        
        # Removes the ```json
        text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\n?```(?:json)?\s*$', '', flags=re.MULTILINE | re.IGNORECASE)

        # Strips whitespace
        text = text.strip()

        # If it's now a valid JSON-ish output, return as str for framework.
        if text.startswith('{') and text.endswith('}'):
            # Ensure the newlines in strings are escaped if needed, but the LLM model usually does this.
            pass

        # Override the response.
        if isinstance(response, dict):
            response['choices'][0]['message']['content'] = text
        else:
            response = text # If str response.

        return response

MODEL_GROQ = 'groq/meta-llama/llama-4-scout-17b-16e-instruct'
MODEL_2 = 'groq/openai/gpt-oss-120b'
MODEL_3 = 'groq/meta-llama/llama-guard-4-12b'
MODEL_GOOGLE = 'gemini/gemini-2.0-flash'

client = StrippingLiteLLMClient(model=MODEL_GOOGLE, temperature=0.1, api_key=os.getenv("GOOGLE_API_KEY"))

# Optional custom curator prompt; used in case default curator prompt fails.
custom_curator_prompt = '''
Progress: {progress}
Stats: {stats}
Reflection: {reflection}
Playbook: {playbook}
Context: {question_context}
Decide what changes to make. Return valid and parseable JSON with delta operations.
'''

generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)
playbook = Playbook()

SEED_BULLETS = [
    "OUTPUT FORMAT: Respond with a raw JSON object ONLY: {\"reasoning\": \"...\", \"bullet_ids\": [...], \"final_answer\": \"RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0\"}. NO Markdown fences ('''json), NO extra text, NO explanations outside JSON.",
    "CRITICAL: In 'final_answer', use EXACTLY: RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0 at end. Ensure a valid and parseable JSON overall.",
    "JSON RULES: Use \\n for line breaks in 'reasoning'. Escape quotes properly. No apostrophes in strings if possible.",
    "CHAIN OF THOUGHT: 1. Quote evidence. 2. Map to rubric level. 3. Justify score. Put in 'reasoning'.",
    "DETERMINISM: Ignore creativity; stick to rubric descriptors verbatim.",
    "RUBRIC BULLETS: Use these for grading.", # rubric_bullets will be added dynamically.
]

for bullet in SEED_BULLETS:
    playbook.add_bullet(bullet, content='essential')