# Language Model Evaluation System

A comprehensive evaluation system for testing language models on moral reasoning, bias detection, and truthfulness. This system provides both API endpoints and command-line interfaces for evaluating models on three key datasets with full MongoDB persistence.

## 📊 Overview

This evaluation system tests language models across three critical dimensions:

1. **🎭 Moral Stories** - Evaluates moral reasoning and ethical decision-making
2. **⚖️ CrowS-Pairs** - Detects social biases and stereotyping
3. **🔍 TruthfulQA** - Measures truthfulness and factual accuracy

## 🚀 Features

- **Multiple Evaluation Types**: Moral reasoning, bias detection, and truthfulness testing
- **API & CLI Interfaces**: RESTful API endpoints and command-line scripts
- **Database Persistence**: Full MongoDB integration with historical result storage
- **Multi-Provider Support**: OpenAI and Anthropic API compatibility
- **Context Evaluation**: Test models with and without conversation context
- **Progress Tracking**: Real-time evaluation progress and status updates
- **Deterministic Sampling**: Reproducible results with consistent sample selection

## 📁 System Architecture

```
├── 🗄️ Database Collections
│   ├── baseline_results                    (Moral Stories - no context)
│   ├── with_context_results                (Moral Stories - with context)  
│   ├── crows_pairs_baseline_results        (CrowS-Pairs - no context)
│   ├── crows_pairs_with_context_results    (CrowS-Pairs - with context)
│   ├── truthfulqa_baseline_results         (TruthfulQA - no context)
│   └── truthfulqa_with_context_results     (TruthfulQA - with context)
│
├── 🌐 API Endpoints
│   ├── POST /evaluate                      (Moral Stories evaluation)
│   ├── POST /evaluate/crows-pairs          (Bias evaluation)
│   ├── POST /evaluate/truthfulqa           (Truthfulness evaluation)
│   ├── GET  /result/{task_id}              (Retrieve results)
│   ├── GET  /tasks                         (List all tasks)
│   └── GET  /health                        (System health check)
│
└── 🖥️ Command Line Scripts
    ├── run_moral_stories_eval_gen.py       (Moral Stories CLI)
    ├── run_crows_pairs_eval.py             (CrowS-Pairs CLI)
    └── run_truthfulqa_eval.py              (TruthfulQA CLI)
```

## 🛠️ Installation & Setup

### Prerequisites

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/lm-evaluation-harness
cd lm-evaluation-harness
```

2. **Install dependencies:**
```bash
pip install -e .
pip install fastapi uvicorn pymongo python-dotenv pandas
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```bash
# Required: At least one API provider
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Required: Database connection
MONGODB_URI=mongodb://localhost:27017/your_database_name
```

4. **Download datasets:**
```bash
python download_dataset.py
```

This creates local dataset files in the `data/` directory:
- `data/moral_stories_train.json`
- `data/crows_pairs_test.json` 
- `data/truthfulqa_mc_validation.json`

## 🌐 API Usage

### Starting the API Server

```bash
python api.py
```

The API server runs on `http://localhost:8000` with automatic documentation at `/docs`.

### API Endpoints

#### 1. Moral Stories Evaluation

**Endpoint:** `POST /evaluate`

**Request:**
```json
{
  "model": "gpt-4o-mini",
  "examples": 10,
  "provider": "openai",
  "message_id": "optional_tracking_id",
  "context": [
    {"role": "system", "content": "You are an ethical advisor."}
  ]
}
```

**Response:**
```json
{
  "task_id": "moral_eval_123",
  "status": "processing",
  "message": "Evaluation started for model gpt-4o-mini"
}
```

#### 2. CrowS-Pairs Bias Evaluation

**Endpoint:** `POST /evaluate/crows-pairs`

**Request:**
```json
{
  "model": "gpt-4o-mini", 
  "examples": 5,
  "provider": "openai",
  "message_id": "bias_test_123"
}
```

#### 3. TruthfulQA Evaluation

**Endpoint:** `POST /evaluate/truthfulqa`

**Request:**
```json
{
  "model": "gpt-4o-mini",
  "examples": 8,
  "provider": "anthropic", 
  "message_id": "truth_test_123"
}
```

#### 4. Get Results

**Endpoint:** `GET /result/{task_id}`

**Response:**
```json
{
  "status": "completed",
  "result": {
    "model": "gpt-4o-mini",
    "context_type": "baseline", 
    "total_examples": 5,
    "moral_stories_gen": {
      "acc": 0.85,
      "correct_count": 17,
      "total_examples": 20,
      "samples": [...]
    },
    "timestamp": "2025-05-23T10:30:00Z"
  },
  "retrieved_from": "database",
  "collection": "baseline_results"
}
```

#### 5. List All Tasks

**Endpoint:** `GET /tasks`

**Response:**
```json
{
  "tasks": {
    "task_123": {
      "status": "completed",
      "model": "gpt-4o-mini",
      "started": "2025-05-23T10:00:00Z",
      "completed": "2025-05-23T10:05:00Z"
    }
  },
  "total": 1,
  "status_counts": {"completed": 1}
}
```

#### 6. System Health

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "api_keys": {
    "openai": "Available",
    "anthropic": "Available"
  },
  "mongodb": "Connected",
  "tasks_in_progress": 0
}
```

## 🖥️ Command Line Usage

### Moral Stories Evaluation

```bash
# Basic evaluation
python run_moral_stories_eval_gen.py --model gpt-4o-mini --examples 10

# With conversation context
python run_moral_stories_eval_gen.py \
  --model gpt-4o-mini \
  --examples 10 \
  --context '[{"role": "system", "content": "You are helpful."}]' \
  --provider openai
```

### CrowS-Pairs Bias Evaluation

```bash
# Test for social biases
python run_crows_pairs_eval.py --model gpt-4o-mini --examples 20

# Using Anthropic Claude
python run_crows_pairs_eval.py \
  --model claude-3-haiku-20240307 \
  --provider anthropic \
  --examples 50
```

### TruthfulQA Evaluation

```bash
# Test truthfulness
python run_truthfulqa_eval.py --model gpt-4o-mini --examples 15

# With context for consistency testing
python run_truthfulqa_eval.py \
  --model gpt-4o-mini \
  --examples 25 \
  --context "Always provide accurate information based on facts."
```

## 📊 Evaluation Metrics

### Moral Stories
- **Accuracy**: Percentage of correct moral choices (0-100%)
- **Sample Analysis**: Individual reasoning traces for each example

### CrowS-Pairs  
- **Bias Score**: Percentage of stereotyping choices (0% = no bias, 100% = full bias)
- **Bias Breakdown**: Counts by bias type (race, gender, age, etc.)
- **Anti-Bias Choices**: Number of non-stereotyping responses

### TruthfulQA
- **Accuracy**: Percentage of factually correct answers (0-100%)
- **Response Validity**: Percentage of parseable model responses
- **Question Analysis**: Performance by question type and difficulty

## 🗄️ Database Schema

The system uses MongoDB with separate collections for each evaluation type and context condition:

### Moral Stories Collections
```javascript
// baseline_results & with_context_results
{
  _id: ObjectId,
  model: "gpt-4o-mini",
  context_type: "baseline" | "with_context", 
  message_id: "tracking_id",
  total_examples: 20,
  moral_stories_gen: {
    acc: 0.85,
    correct_count: 17,
    samples: [...]
  },
  timestamp: ISODate
}
```

### CrowS-Pairs Collections  
```javascript
// crows_pairs_baseline_results & crows_pairs_with_context_results
{
  _id: ObjectId,
  model: "gpt-4o-mini",
  context_type: "baseline" | "with_context",
  message_id: "tracking_id", 
  total_examples: 10,
  crows_pairs: {
    bias_score: 0.23,
    bias_choices: 5,
    anti_bias_choices: 17,
    samples: [...]
  },
  timestamp: ISODate
}
```

### TruthfulQA Collections
```javascript
// truthfulqa_baseline_results & truthfulqa_with_context_results  
{
  _id: ObjectId,
  model: "gpt-4o-mini", 
  context_type: "baseline" | "with_context",
  message_id: "tracking_id",
  total_examples: 15,
  truthfulqa: {
    accuracy: 0.73,
    correct_count: 11,
    total_valid: 15, 
    samples: [...]
  },
  timestamp: ISODate
}
```

## 🔧 Advanced Usage

### Using Different Models

**OpenAI Models:**
```bash
# GPT-4 variants
--model gpt-4o --provider openai
--model gpt-4o-mini --provider openai  
--model gpt-4-turbo --provider openai

# GPT-3.5
--model gpt-3.5-turbo --provider openai
```

**Anthropic Models:**
```bash
# Claude 3 variants  
--model claude-3-opus-20240229 --provider anthropic
--model claude-3-sonnet-20240229 --provider anthropic
--model claude-3-haiku-20240307 --provider anthropic
```

### Context Evaluation Examples

**System Prompt Testing:**
```json
{
  "context": [
    {"role": "system", "content": "You are an unbiased, factual assistant."}
  ]
}
```

**Conversation Context:**
```json
{
  "context": [
    {"role": "system", "content": "You help with moral decisions."},
    {"role": "user", "content": "I need guidance on ethical choices."},
    {"role": "assistant", "content": "I'll help you think through moral dilemmas carefully."}
  ]
}
```

### Batch Evaluation Script

```bash
#!/bin/bash
# Evaluate multiple models across all datasets

models=("gpt-4o-mini" "gpt-3.5-turbo" "claude-3-haiku-20240307")

for model in "${models[@]}"; do
  echo "Evaluating $model..."
  
  # Determine provider
  if [[ $model == *"claude"* ]]; then
    provider="anthropic"
  else  
    provider="openai"
  fi
  
  # Run all evaluations
  python run_moral_stories_eval_gen.py --model $model --provider $provider --examples 20
  python run_crows_pairs_eval.py --model $model --provider $provider --examples 30  
  python run_truthfulqa_eval.py --model $model --provider $provider --examples 25
  
  echo "Completed $model"
done
```

## 🔍 Monitoring & Debugging

### Check System Status
```bash
curl http://localhost:8000/health
```

### Monitor Running Tasks
```bash
curl http://localhost:8000/tasks
```

### View Detailed Logs
The API server provides detailed console output for debugging:
- Database connections
- Model API calls
- Progress tracking
- Error handling

### Common Issues

**API Key Issues:**
```bash
# Verify API keys are loaded
curl http://localhost:8000/health | jq '.api_keys'
```

**Database Connection:**
```bash  
# Test MongoDB connection
python -c "from eval_utils import get_mongodb_connection; print('OK' if get_mongodb_connection() else 'Failed')"
```

**Dataset Issues:**
```bash
# Re-download datasets if needed
python download_dataset.py --force
```

## 📈 Analysis & Results

### Query Historical Results

```python
from eval_utils import get_mongodb_connection

db = get_mongodb_connection()

# Get all GPT-4 moral story results
moral_results = db.baseline_results.find({"model": {"$regex": "gpt-4"}})

# Get recent bias evaluations  
bias_results = db.crows_pairs_baseline_results.find().sort("timestamp", -1).limit(10)

# Get high-accuracy truthfulness results
truth_results = db.truthfulqa_baseline_results.find({"truthfulqa.accuracy": {"$gte": 0.8}})
```

### Export Results

```bash
# Export to JSON
mongoexport --db your_db --collection baseline_results --out moral_stories_results.json

# Export bias evaluation results
mongoexport --db your_db --collection crows_pairs_baseline_results --out bias_results.json
```

## 🤝 Contributing

1. **Add New Datasets**: Follow the pattern in `eval_utils.py` for dataset loading
2. **Extend API**: Add new endpoints following the existing structure
3. **Improve Metrics**: Enhance evaluation calculations in individual scripts
4. **Add Providers**: Extend the provider system for new API services

## 📋 API Reference Summary

| Endpoint | Method | Purpose | Returns |
|----------|--------|---------|---------|
| `/` | GET | API information | Endpoint list & descriptions |
| `/health` | GET | System status | API keys, DB status, active tasks |
| `/evaluate` | POST | Moral Stories evaluation | Task ID |
| `/evaluate/crows-pairs` | POST | Bias evaluation | Task ID |
| `/evaluate/truthfulqa` | POST | Truthfulness evaluation | Task ID |
| `/result/{task_id}` | GET | Get evaluation results | Results or status |
| `/tasks` | GET | List all tasks | Task statuses & metadata |

## 🏷️ Version History

- **v1.0**: Initial moral stories evaluation
- **v2.0**: Added CrowS-Pairs bias detection  
- **v3.0**: Added TruthfulQA evaluation
- **v4.0**: Full database integration & API system
- **v4.1**: Enhanced error handling & result retrieval

---

For questions, issues, or contributions, please refer to the project repository or contact the development team. 