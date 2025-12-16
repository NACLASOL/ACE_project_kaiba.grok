import os
from pathlib import Path

# === BASE DIRECTORIES ===
BASE_DIR = Path(".").resolve()

# Directory where the parsed PDFs are stored (rubric, examples)
PARSE_DIR = Path(os.getenv("ACE_PARSE_DIR", BASE_DIR / "parse")).resolve()

# Logs root.
LOG_DIR = Path(os.getenv("ACE_LOG_DIR", BASE_DIR / "logs")).resolve()

# Epochs dir under logs (used by adapt.py).
EPOCHS_DIR = Path(os.getenv("ACE_EPOCHS_DIR", LOG_DIR / "epochs")).resolve()

# Ensure the core directories exist.
LOG_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS_DIR.mkdir(parents=True, exist_ok=True)


# === FILES ===
RUBRIC_FILE = Path(
    os.getenv("ACE_RUBRIC_FILE", PARSE_DIR / "internship.enhance_b2_article_writing_rubric_structured.pdf")
).resolve()

# Student examples PDF.
EXAMPLES_FILE = Path(
    os.getenv("ACE_EXAMPLES_FILE", PARSE_DIR / "testing_batch_30_examples.pdf")
).resolve()

# JSON summarising adaptation loop.
ADAPTATION_SUMMARY_FILE = LOG_DIR / "adaptation_summary.json"

# Latest playbook file.
LATEST_PLAYBOOK_FILE = LOG_DIR / "latest_playbook.json"

# Parsing debug log.
PARSE_DEBUG_LOG = LOG_DIR / "parse_debug.log"

# Grade logs.
GRADES_LOG_FILE = LOG_DIR / "grades.log"

# Epoch failure log for adaptatio.
EPOCH_FAILURE_LOG = LOG_DIR / "epoch_failure.log"

# JSON failures from ACE roles.
JSON_FAILURE_LOG = LOG_DIR / "json_failures.log"

# === TASK PROMPT ===
TASK_PROMPT = """
You have just seen the following advertisement in the university magazine:
Share with us what you think about the importance of physically attending classes in-person for university students instead of online classes. We're looking for articles about the benefits of studying in a classroom with a teacher. The best article will be published in our university magazine and the winner will receive a €200 gift card.
You have decided to contribute, Write an ARTICLE in which you:
• explain why physically attending classes can improve learning.
• mention why attending class at a university also provides the student with other
resources and facilities.
• suggest ways in which teachers can encourage students to go to class.
Give your article a title. Write your ARTICLE in 180-220 words.
"""

# === OPIK CONFIGURATION ===
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME")