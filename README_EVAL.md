# Simple Moral Stories Evaluation

This is a simple script to evaluate language models on the "Moral Stories" task from the Language Model Evaluation Harness. The script runs an evaluation on a limited set of examples (default: 5) to quickly test how a model performs on moral reasoning.

## About the Moral Stories Task

Moral Stories is a dataset that evaluates a model's ability to distinguish between moral and immoral actions in specific situations. Each example consists of:

- A norm (social guideline)
- A situation (setting of the story)
- An intention (a goal one character wants to achieve)
- Two choices: a moral action and an immoral action

The model is asked to choose between the two actions based on the given context.

## Usage

### Prerequisites

Make sure you have installed the Language Model Evaluation Harness:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Running the Evaluation

To run the evaluation with default settings (using the gpt2 model on CPU):

```bash
python run_moral_stories_eval.py
```

### Custom Settings

You can customize the evaluation with the following arguments:

- `--model`: Model type to use (e.g., "hf", "openai-completions", "vllm")
- `--model_args`: Arguments for the model (e.g., "pretrained=gpt2" or "model=text-davinci-003")
- `--device`: Device to run the model on (e.g., "cpu", "cuda:0")

Examples:

#### Using a Hugging Face model on GPU:

```bash
python run_moral_stories_eval.py --model hf --model_args pretrained=gpt2-medium --device cuda:0
```

#### Using an OpenAI model:

```bash
export OPENAI_API_KEY=your_api_key_here
python run_moral_stories_eval.py --model openai-completions --model_args model=text-davinci-003
```

#### Using vLLM with a larger model:

```bash
python run_moral_stories_eval.py --model vllm --model_args pretrained=meta-llama/Llama-2-7b-hf --device cuda:0
```

## Results

The script will output the results to the console and save them to a file named `moral_stories_results.json`. The results include:

- Accuracy: Percentage of examples where the model correctly preferred the moral action
- Normalized Accuracy: Similar to accuracy but normalized for model biases
- Sample evaluations: The queries, choices, and model responses for the evaluated examples 