# CoT Evaluation for Video Reasoning

This repository contains the official evaluation codebase for the NeurIPS submission on CoT Evaluation for Video Reasoning.

## Overview

The codebase provides tools for evaluating model-generated reasoning (Chain-of-Thought) across multiple dimensions including correctness, hallucination, and logical divergence.

### Core Scripts

- **`eval_models.py`**: Main inference and evaluation script for comparing multiple API-based models (ChatGPT, Claude, Gemini, Grok). Supports Socratic sequential-chunk processing for long videos.
- **`run_judges.py`**: Multi-judge evaluation framework. It utilizes advanced LLMs as judges to rate model outputs across several facets:
  - `divergence`: Measures deviation from reference reasoning.
  - `hallucination`: Detects unsupported or fabricated details.
  - `finalanswer`: Assesses the semantic correctness of the final conclusion.
- **`run_token_conciseness.py`**: A verifiable evaluation tool that calculates conciseness scores based on token counts (using `tiktoken`), providing a deterministic metric for reasoning efficiency.
- **`likert_score_analysis.py`**: Statistical analysis tool for judge agreement. Computes inter-rater reliability (Cohen's Kappa and Fleiss' Kappa) across different judge models and evaluated reasoning facets.

## Dependencies

- **Python**: 3.10+
- **Packages**: `openai`, `anthropic`, `google-genai`, `tiktoken`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `loguru`, `tqdm`, `tenacity`.
- **System Tools**: `ffmpeg` and `ffprobe` are required for video processing and frame extraction.

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Model Evaluation
Run the main evaluation script with your dataset:
```bash
python eval_models.py \
    --dataset data/your_dataset.jsonl \
    --output results/eval_results.jsonl \
    --use_socratic
```

### 2. Multi-Judge Scoring
To evaluate specific facets of the generated reasoning:
```bash
python run_judges.py \
    --data_path ./results/ \
    --output_dir ./scoring_results/ \
    --facets finalanswer hallucination divergence
```

### 3. Token-based Conciseness
To compute deterministic conciseness scores:
```bash
python run_token_conciseness.py \
    --data_path ./results/ \
    --output_dir ./conciseness_results/
```

### 4. Statistical Analysis
To analyze inter-judge agreement and score distributions:
```bash
python likert_score_analysis.py \
    --base_dirs ./scoring_results/ \
    --output_dir ./analysis_results/ \
    --combined_kappa
```

## Environment Variables

Ensure the following API keys are set in your environment:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY` (for Grok)

## License
This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (see `LICENSE.txt`).
