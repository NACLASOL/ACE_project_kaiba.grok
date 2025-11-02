import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook
from parse_data import SEED_BULLETS


load_dotenv()

MODEL_GROQ = 'groq/meta-llama/llama-4-scout-17b-16e-instruct'
MODEL_2 = 'groq/meta-llama/llama-prompt-guard-2-22m'
MODEL_3 = 'groq/meta-llama/llama-guard-4-12b'
MODEL_GOOGLE = 'gemini/gemini-2.0-flash'

client = LiteLLMClient(model=MODEL_GOOGLE, temperature=0.1, api_key=os.getenv("GOOGLE_API_KEY"))

generator = Generator(client)
reflector = Reflector(client)
curator = Curator(client)
playbook = Playbook()

for bullet in SEED_BULLETS:
    playbook.add_bullet(bullet, content='essential')