# ACE Implementation for CEFR B2 English Article Corrections.

## Prerequisites.

- Python 3.11 or higher.
- An API key for Google or Groq.

## Quick-start guide.

### Step 1: Create you virtual environment.
Create your virtual environment:

```bash
# For Windows.
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv <your-venv-name>
```


### Step 2: Download dependencies.
Install the repository dependendies using the `requirements.txt` file:

```bash
# For Windows.
pip install -r .\requirements.txt
```


### Step 3: Set up your environment.

Create a `.env` file with your API key:

```bash
# For Google.
GOOGLE_API_KEY=your-key-here

# For Groq.
GROQ_API_KEY=your-key-here
```

### Step 4: Run the `adapt.py` file.
Create you first Playbook with the `adapt.py` file:

```bash
python adapt.py
```

You now have an AI updated Playbook in `logs\latest_playbook.json`.

### Additional tools.
Execute the `viz.py` file to visualize the Plabook Adaptation progress.
```bash
python viz.py
```

Run the Article Grader application using Streamlit.
```bash
# TO START:
streamlit run grader_ui.py

# TO STOP: `ctrl + c` inside terminal. Then close the application tab in your browser.
```

## File functionality.

- `adapt.py`:
Main adaptation process. Creates an AI updated Notebook.

- `grade_article.py`:
Grades a single CEFR B2 exam article using the latest Playbook adaptation.

- `grader_ui.py`:
Basic Streamlit application to grade student articles.

- `parse_data.py`:
Extracts example-article information from a PDF file.

- `upv_grader`:
Configures the LLM's used by the `adapt.py` file, initializes the main ACE roles, and creates initial Playbook seeds.

- `viz.py`:
Graphic tool to visualize the evolution of the Playbook adaptations.

Made By: Nicholas Clancy Soler.