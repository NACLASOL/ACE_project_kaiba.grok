# ACE Implementation for CEFR B2 English Article Corrections.

## Prerequisites.

- Python 3.11 or higher.
- An API key for Google or Groq.

## Quick-start guide.

Clone this repository using:
```bash
git clone repository-link-here
```
Then proceed as follows inside the repository folder.

### Step 1: Create a virtual environment.
Create and activate your virtual environment:

```bash
# For Windows.
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv <your-venv-name>
# Activate.
\<your-venv-name>\Scripts\Activate.ps1

# For MacOS
python3 -m venv <your-venv-name>
# Activate.
source <your-venv-name>/bin/activate
```


### Step 2: Download dependencies.
Install the repository dependendies using the `requirements.txt` file:

```bash
# For Windows.
pip install -r .\requirements.txt

# For MacOS.
pip install -r requirements.txt
```


### Step 3: Set up your environment.

Create a `.env` file with your API key:

```python
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

## Using Opik Tracking.

Add the following to your `.env` file:

```python
# OPIK TRACKING.
OPIK_PROJECT_NAME=your-project-name-here 
COMET_API_KEY=your-key-here
```
Opik should be installed using the project `requirements.txt` file.
You **may** need to configure your Opik client in your terminal:

```bash
opik configure
```

When running any of the following files: `adapt.py`, `grade_article.py`, `grader_ui.py`; your traces will be logged to your Opik account, as configured during the Opik configuration.

## File functionality.

- `adapt.py`:
Main adaptation process. Uses the main ACE roles to create an AI updated ACE Notebook.

- `grade_article.py`:
This file grades a single CEFR B2 exam article using the latest Playbook adaptation.

- `grader_ui.py`:
Basic Streamlit application to grade student articles individually or in batches.

- `parse_data.py`:
Extracts the CEFR B2 English Article Writing Rubric and student article examples with their corresponding scores from their corresponding PDF files.

- `upv_grader`:
Configures the LLM's used by the `adapt.py` file, initializes the main ACE roles, and creates the initial Playbook seeds.

- `viz.py`:
Graphic tool to visualize the evolution of the Playbook adaptations, including the number of delta operations (changes to the Playbook) performed in one epoch (iteration), and the number of mismatches (differences between AI score and human score per article example).

Made By: Nicholas Clancy Soler.