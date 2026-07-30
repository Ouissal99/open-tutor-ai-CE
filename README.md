# Open TutorAI Agentic Tool Interaction Manager

This branch contains the implementation and evaluation code for a centralized Tool Interaction Manager developed for Open TutorAI.

The manager coordinates task analysis, planning, tool selection, execution, validation, failure recovery, output packaging, and interaction trace logging.

## Main Code

- `ai/agentic/tool_interaction/` — centralized tool-interaction workflow
- `ai/agentic/tools/` — executable tools
- `ai/agentic/memory/` — memory and trace components
- `ai/agentic/tutoring/` — personalized tutoring workflow
- `evaluation/` — evaluation runners, dataset, and results

Implemented tools include `CalculatorTool`, `MatrixComputationTool`, `RAGTool`, `TraceSearchTool`, `CodeSandboxTool`, and `VisualMatrixTool`.

## Requirements

- Python 3.11
- Git
- An OpenAI-compatible model provider
- Provider credentials stored locally in `.env`

## Installation

Clone the repository and open the article branch:

    git clone https://github.com/Ouissal99/open-tutor-ai-CE.git
    cd open-tutor-ai-CE
    git checkout article/align-article-version

Create and activate a Python environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install the project:

    python -m pip install --upgrade pip
    python -m pip install -e .

Create the local environment file:

    cp .env.example .env

Add the required provider configuration to `.env`. Never commit `.env` or API credentials.

For development and testing dependencies:

    python -m pip install -r requirements-ci.txt

## Evaluation

The 100-case dataset is located at:

    evaluation/test_cases/test_cases_100_article.csv

Run the Full Manager:

    python evaluation/run_full_manager_article_dataset.py \
      --dataset evaluation/test_cases/test_cases_100_article.csv \
      --output-dir evaluation/local_results/full_manager

Run the Direct LLM baseline:

    python evaluation/baselines/direct_llm/run_direct_llm_article_100.py \
      --dataset evaluation/test_cases/test_cases_100_article.csv \
      --output-dir evaluation/local_results/direct_llm

Run the Basic Tool Use baseline:

    python evaluation/baselines/basic_tool_use/run_basic_tool_use_article_100.py \
      --dataset evaluation/test_cases/test_cases_100_article.csv \
      --output-dir evaluation/local_results/basic_tool_use

Run the Manager Without Recovery baseline:

    python evaluation/baselines/manager_without_recovery/run_manager_without_recovery_article_100.py \
      --dataset evaluation/test_cases/test_cases_100_article.csv \
      --output-dir evaluation/local_results/manager_without_recovery

Official evaluation artifacts are stored under:

    evaluation/final_results_100/

Use a new output directory for new runs. Do not overwrite the official evaluation artifacts.
