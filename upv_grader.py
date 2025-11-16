import os
import json
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook
from parse_data import rubric_bullets


load_dotenv()

# THIS FUNCTION DOES NOT WORK.
class StrippingLiteLLMClient(LiteLLMClient):
    '''Custom LiteLLMClient that strips Markdown fences from LLM response before returning'''
    def query(self, prompt, **kwargs):
        response = super().query(prompt, **kwargs)

        # ACE expects Response with .text; base does this.
        if not hasattr(response, "text"):
            # LiteLLM sometimes returns a dict with ".choices"
            if isinstance(response, dict) and "choices" in response:
                text = response["choices"][0]["message"]["content"]
            else:
                text = str(response)
            # Fake Response.
            response = type('FakeResponse', (), {'text':text})()
        
        # Prints original .text for debug.
        print("Original text[:40]", response.text[:50])

        # --- STRIP FENCES ----
        raw = response.text.strip()

        # Removes the ```json.
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.IGNORECASE)

        # Removes occasional single-backtick fences.
        raw = re.sub(r"^`{1,3}\s*", "", raw)
        raw = re.sub(r"\s*`{1,3}$", "", raw)

        raw = raw.strip()

        # Fallback if empty.
        if not raw:
            raw = '{}'
        
        response.text = raw

        # Print stripped .text for debug.
        print("Stripped text:", response.text)

        return response

MODEL_GROQ = 'groq/meta-llama/llama-4-scout-17b-16e-instruct'
MODEL_2 = 'groq/llama-3.3-70b-versatile'
MODEL_3 = 'groq/meta-llama/llama-guard-4-12b'
MODEL_GOOGLE = 'gemini/gemini-2.0-flash'

client = StrippingLiteLLMClient(model=MODEL_GROQ, api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)

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
    "OUTPUT FORMAT: Respond with a raw JSON object ONLY: {\"reasoning\": \"...\", \"bullet_ids\": [...], \"final_answer\": \"RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0\"}. NO Markdown fences (```json), NO extra text, NO explanations outside JSON.",
    "CRITICAL: In 'final_answer', use EXACTLY: RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0 at end. Ensure a valid and parseable JSON overall.",
    "JSON RULES: Use \\n for line breaks in 'reasoning'. Escape quotes properly. No apostrophes in strings.",
    "CHAIN OF THOUGHT: 1. Quote evidence. 2. Map to rubric level. 3. Justify score. Put in 'reasoning'.",
    "DETERMINISM: Ignore creativity; stick to rubric descriptors verbatim.",
    "RUBRIC BULLETS: Use these for grading.", # rubric_bullets will be added dynamically.
]

for bullet in SEED_BULLETS:
    playbook.add_bullet(bullet, content='essential')