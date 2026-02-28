# ACE Implementation for CEFR B2 English Article Grading

An AI-powered grading system for CEFR B2 English articles using **Agentic Context Engineering (ACE)**. This system uses iterative playbook refinement to align AI grading with human expert scores.

## 🎯 Features

- **Adaptive Grading**: Uses ACE framework roles (Generator, Reflector, Curator) to evolve grading strategies
- **Checkpoint System**: Resume interrupted adaptation runs without losing progress
- **Pruning System**: Automatically prunes the ACE playbook during the adaptation using select strategies
- **Batch Processing**: Grade multiple articles simultaneously via Streamlit UI
- **Opik Tracking**: Full observability with execution traces and metadata
- **Visualization Tools**: Track playbook evolution and mismatch reduction across epochs
- **Configuration Validation**: Comprehensive startup checks for missing dependencies

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **API Access**: Google Gemini, Groq API key, or other LLM API's
- **Opik Account**: (Optional) For execution tracking and observability
- **PDF Files**: Rubric and student examples (see Project Structure)

## 🚀 Quick Start Guide

### 1. Clone Repository

```bash
git clone https://github.com/NACLASOL/ACE_project_kaiba.grok
cd ACE_project_kaiba.grok
```

### 2. Create Virtual Environment

**Windows:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# AI MODEL API KEYS.
GOOGLE_API_KEY=
GROQ_API_KEY=
UPV_API_KEY=

# OPIK TRACKING.
OPIK_PROJECT_NAME=
COMET_API_KEY=
OPIK_TRACK_DISABLE=True

# CONFIGURATION
VIZ_FORMAT=png

# PROJECT DIRECTORIES.
ACE_PARSE_DIR=./parse
ACE_LOG_DIR=./logs
ACE_EPOCHS_DIR=./logs/epochs

# PROJECT FILES.
ACE_RUBRIC_FILE=parse/internship.enhance_b2_article_writing_rubric_structured.pdf
ACE_EXAMPLES_FILE=parse/debug_batch_4_examples.pdf

# PRUNING CONFIGURATION
PRUNING_ENABLED=True
PRUNING_STRATEGY=hybrid
PRUNING_SIMILARITY_THRESHOLD=0.70
PRUNING_MIN_USAGE_COUNT=3
PRUNING_AGE_THRESHOLD_EPOCHS=2
PRUNING_MIN_PLAYBOOK_SIZE=10
PRUNING_MAX_REMOVE_PER_EPOCH=30
PRUNING_DRY_RUN=False
```

### 5. Prepare Data Files

Place your rubric and examples PDFs in the `parse/` directory:
```
parse/
├── internship.enhance_b2_article_writing_rubric_structured.pdf
├── internship.b2_writing_examples_scores.pdf
└── training_batch_70_examples.pdf
...
```

### 6. Validate Configuration

```bash
python constants.py
```

This validates all configuration settings and reports any issues.

### 7. Run Adaptation Process

**Standard Mode:**
```bash
python adapt.py
```

**Checkpoint Mode (Resume on Interrupt):**
```bash
python adapt.py
# Press Ctrl+C to save progress
# Run again to resume from last completed epoch
```

**Custom Epochs:**
```bash
python adapt.py 5  # Run 5 epochs instead of default
```

Upon the completion of the adaptation process, the evolved playbook will be saved to `logs/latest_playbook.json`.

## 📊 Usage Examples

### Visualize Adaptation Progress

```bash
python viz.py
```

Generates plots showing:
- Average mismatch vs. playbook size over epochs
- Delta operations (ADD/MODIFY/DELETE) per epoch
- Playbook growth trajectory

### Grade Individual Article

```bash
python grade_article.py
```

Edit the example at the bottom of the file or import `grade_article()` function.

### Launch Streamlit UI

```bash
streamlit run grader_ui.py
```

**Features:**
- Single article grading with detailed justification
- Batch CSV upload (semicolon-separated)
- Download results as JSON
- Average statistics for batch grading

**CSV Format for Batch Grading:**
```csv
title;text
"Article Title 1";"Full article text here..."
"Article Title 2";"Another article text..."
```

## 📁 Project Structure

```
ACE_project_kaiba.grok/
├── adapt.py              # Main adaptation loop with ACE framework
├── grade_article.py      # Production grading with evolved playbook
├── grader_ui.py          # Streamlit web interface
├── parse_data.py         # PDF extraction (rubric + examples)
├── upv_grader.py         # LLM client and ACE component initialization
├── checkpoint.py         # Checkpoint/resume system with signal handling
├── prune.py              # Adaptation playbook pruning using strategies
├── viz.py                # Visualization tools for adaptation metrics
├── constants.py          # Configuration management and validation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── parse/                # PDF files directory
│   ├── internship.enhance_b2_article_writing_rubric_structured.pdf
│   ├── internship.b2_writing_examples_scores.pdf
│   └── ...
└── logs/                 # Generated output directory
    ├── latest_playbook.json
    ├── adaptation_checkpoint.json
    ├── adaptation_summary.json
    ├── grades.log
    ├── parse_debug.log
    ├── samples/
    │   └── train_samples.json
    └── epochs/
        ├── epoch-00.json
        ├── epoch-01.json
        └── ...
```

## 🔧 Configuration Details

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPIK_PROJECT_NAME` | ❌ No | - | Project name for Opik tracking |
| `OTHER_API_KEY` | ⚠️ One required | - | LLM API key |
| `GOOGLE_API_KEY` | ⚠️ One required | - | Google Gemini API key |
| `GROQ_API_KEY` | ⚠️ One required | - | Groq API key |
| `COMET_API_KEY` | ❌ No | - | Opik/Comet ML API key |
| `OPIK_TRACK_DISABLE` | ✅ Yes | False | Enable Opik trace logging |
| `ACE_LOG_DIR` | ✅ Yes | `./logs` | Output directory |
| `ACE_PARSE_DIR` | ✅ Yes | `./parse` | PDF files directory |
| `CHECKPOINT_ENABLED` | ✅ Yes | `True` | Enable checkpoint/resume |
| `DEBUG_ADAPT` | ❌ No | `False` | Enable debug logging |
| PRUNING VARIABLES | ✅ Yes | Controls pruning system |

### LLM Model Configuration

Edit `upv_grader.py` to change the model:

```python
# Available options:
MODEL_YOUR-MODEL = ...
MODEL_GROQ = 'groq/meta-llama/llama-4-scout-17b-16e-instruct'
MODEL_GOOGLE = 'gemini/gemini-2.5-flash'
```

## 🎓 How It Works

### ACE Framework Components

1. **Generator**: Produces initial article grades using current playbook
2. **Reflector**: Analyzes mismatches between AI and human scores
3. **Curator**: Proposes playbook modifications (ADD/MODIFY/DELETE operations)
4. **Playbook**: Evolving knowledge base of grading strategies

### Adaptation Loop

```
For each epoch:
  For each training sample:
    1. Generate scores using current playbook
    2. Compare with human ground truth
    3. Reflect on mismatches
    4. Curate playbook improvements
    5. Apply deltas to playbook
  
  Perform playbook pruning
  Save epoch checkpoint
  Log metrics (mismatch, playbook size, deltas, bullet usage)
```

### Checkpoint System

- **Automatic saving** after each completed epoch
- **Interrupt handling**: Press `Ctrl+C` once to save and exit safely
- **Resume capability**: Run `adapt.py` again to continue from last checkpoint
- **Atomic writes**: Uses `os.fsync()` to prevent data corruption

### Pruning System
- **Automatic pruning** after each epoch
- **Pruning strategy**: Choose your preferred pruning strategy
- **Strategy parameters**: Tune your chosen strategy parameters
- **Review changes**: Analyze changes during/after the adaptation

## 📈 Output Files

| File | Description |
|------|-------------|
| `latest_playbook.json` | Evolved playbook after adaptation |
| `adaptation_summary.json` | Aggregated metrics across all epochs |
| `adaptation_checkpoint.json` | Resume state (deleted on completion) |
| `epochs/epoch-XX.json` | Per-epoch metrics (mismatch, deltas, size) |
| `grades.log` | Production grading audit log |
| `parse_debug.log` | PDF extraction debugging info |
| `samples/train_samples.json` | Extracted training examples |
| `bullet_metadata` | Metadata por playbook bullets |
| `pruning_log` | Pruning operations performed |
| `usage_log` | Per-epoch bullets used |
| `usage_stats` | Usage data for playbook bullets |

## 🔍 Troubleshooting

### Configuration Errors

**Error: `OPIK_PROJECT_NAME` missing**
```bash
# Add to .env file:
OPIK_PROJECT_NAME=my-project-name
```

**Error: Rubric/Examples PDF not found**
```bash
# Ensure files exist:
ls parse/*.pdf

# Or set custom paths in .env:
ACE_RUBRIC_FILE=./parse/your_rubric.pdf
ACE_EXAMPLES_FILE=./parse/your_examples.pdf
```

### Runtime Issues

**Empty LLM Response (Token Limit Exceeded)**
- Symptom: `TypeError: empty response` or `LLM API failure`
- Solution: Reduce playbook size or use a model with larger context window

**Checkpoint Corruption**
```bash
# Delete corrupted checkpoint and restart:
rm logs/adaptation_checkpoint.json
python adapt.py
```

**Score Extraction Failure**
- Check that LLM output follows format: `RESULTS,TA:X.0,CC:X.0,GR:X.0,LR:X.0,OWP:X.0`
- Review `logs/epoch_failure.log` for debugging details

### Opik Tracking Issues

```bash
# Reconfigure Opik client:
opik configure

# Test connection:
python -c "from opik import track; print('Opik configured successfully')"
```

## 📊 Evaluation Metrics

The system tracks:
- **Average Mismatch**: Mean absolute error between AI and human scores
- **Playbook Size**: Number of strategic bullets in playbook
- **Delta Operations**: ADD/MODIFY/DELETE counts per epoch
- **Score Categories**: TA (Task Achievement), CC (Cohesion), GR (Grammar), LR (Lexical), OWP (Overall)
- **Pruning Operations**: Pruned bullets from the playbook after each epoch
- **Bullet Usage Log**: Playbook bullet usage per epoch
-**Bullet Usage Data**: Bullet metadata

## 🤝 Contributing

This is an academic research project. For collaboration opportunities or questions:

**Author**: Nicholas Clancy Soler  
**Email**: naclasol@masters.upv.es  
**Institution**: Universitat Politècnica de València (UPV)  
**Repository**: [https://github.com/NACLASOL/ACE_project_kaiba.grok](https://github.com/NACLASOL/ACE_project_kaiba.grok)

## 📄 License

Academic research project - please contact author for usage permissions.

## 🙏 Acknowledgments

- **ACE Framework**: Agentic Context Engineering methodology
- **Opik**: Execution tracing and observability platform
- **LiteLLM**: Unified LLM API interface
- **Streamlit**: Web application framework

---

**Last Updated**: February 2026
