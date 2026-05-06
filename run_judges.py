# Single sample evaluation test with configurable judge model
from src.xVerify.model import Model
from src.xVerify.eval import Evaluator

# Import specific evaluators from the same file if needed, or define them here
# (Assuming the classes defined below are what we use)

import json
import os
import datetime
import pandas as pd
from collections import OrderedDict, defaultdict
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# ========================
# JUDGE MODEL CONFIGURATION
# ========================
JUDGE_CONFIGS = {
    "grok-4-1-fast": {
        "base_url": "https://api.x.ai/v1",
        "api_key": os.environ.get("XAI_API_KEY", "<XAI API Key>")
    },
    # "gemini-2.5-flash": {
    #     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    #     "api_key": os.environ.get("GOOGLE_API_KEY", "<GOOGLE API Key>")
    # },
    "claude-haiku-4-5-20251001": {
        "base_url": "https://api.anthropic.com/v1", # Placeholder for OpenAI-compatible proxy
        "api_key": os.environ.get("ANTHROPIC_API_KEY", "<ANTHROPIC API Key>")
    },
    "gpt-5-nano": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY", "<OPENAI API Key>")
    }
}

# ========================
# CUSTOM PROMPT (OPTIONAL)
# ========================
# ... (Same evaluator classes as before: DivergenceEvaluator, HallucinationEvaluator, FinalanswerEvaluator)
# (Keeping the user's provided classes)

class DivergenceEvaluator(Evaluator):
    def __init__(self, model: Model, process_num: int = 5):
        super().__init__(model, process_num)
        # prompt definition moved here to ensure it's used
        self.prompt = '''You are an expert evaluator trained to assess how closely a model's reasoning aligns with a reference reasoning process in text-based tasks.

You will receive:
- A **question** (text only),
- A **model-generated chain of thought (output)**,
- A **ground-truth chain of thought (answer)**.

Your task is to evaluate the *divergence* between the model's reasoning (**output**) and the reference reasoning (**answer**) — that is, how much the model's reasoning process deviates in logic, structure, and interpretation of the question.

**Important:** *Ignore all other issues (such as hallucination) entirely. This evaluation measures **only divergence**, independently from other facets.*

---

### Evaluation Guidelines

Assign a **divergence rating** on a Likert scale from **1 to 5**:

- **5** → Identical or nearly identical reasoning (minimal divergence).  
- **4** → Slight divergence in focus or emphasis, but overall similar reasoning flow.  
- **3** → Noticeable differences in reasoning steps or logic, though partially aligned.  
- **2** → Major divergence in reasoning structure, logic, or conclusions.  
- **1** → Completely unrelated or incorrect reasoning.

When rating, focus on:
- Alignment of logical steps and intermediate conclusions.  
- Consistency in interpretation of the question.  
- Similarity of causal or deductive reasoning structure.  
- *Ignore verbosity, formatting, or hallucination issues.*

---

### Input

**Question:**  
"""{question}"""

**Model Output (Predicted Chain of Thought):**  
"""{output}"""

**Ground Truth Chain of Thought:**  
"""{answer}"""

---

### Output Format

The **first line** of your response must contain only the JSON rating in the following format:  
{{"rating": <integer between 1 and 5>}}

Subsequent lines should provide a **brief textual justification** explaining why you assigned that rating.

Example:  
{{"rating": 3}}  
The model's reasoning partially aligns with the ground truth but introduces different intermediate steps and omits key logical links.
'''
        
    def stat_results(self, results: list[dict]) -> dict:
        valid_ratings = set(range(1, 6))
        valid_num = 0
        rating_freqs = defaultdict(int)
        total_rating = 0
        for item in results:
            judgement = item.get(f'{self.model_name}_judgment_result', "")
            if not judgement:
                item['judge_valid'] = 'False'
                continue
                
            try:
                # Find the first line that looks like JSON (starts with '{'),
                # tolerating leading newlines, markdown fences, etc.
                lines = judgement.splitlines()
                json_line_idx = next(
                    (i for i, l in enumerate(lines) if l.strip().startswith('{')), None
                )
                if json_line_idx is None:
                    item['judge_valid'] = 'False'
                    continue

                rating_json = lines[json_line_idx].strip()
                justification = '\n'.join(lines[json_line_idx + 1:]).strip()
                rating_data = json.loads(rating_json)
                rating = rating_data.get("rating", -1)
                item["reasoning"] = justification

                if rating in valid_ratings:
                    valid_num += 1
                    item['judge_valid'] = 'True'
                    rating_freqs[f"Num_rating={rating}"] += 1
                    total_rating += rating
                else:
                    item['judge_valid'] = 'False'
            except Exception as e:
                logger.error(f"Error parsing judgment: {e}")
                item['judge_valid'] = 'False'
        
        stats = {
            "Valid_num": valid_num,
            "Average": total_rating / valid_num if valid_num > 0 else 0
        }
        for k in range(1, 6):
            stats[f"Num_rating={k}"] = rating_freqs[f"Num_rating={k}"]
            
        return stats

    SAVE_CHUNK_SIZE = 1000  # Save partial results every N entries (for incremental disk saves)

    def batch_gen(self, data, data_name):
        """
        Override of Evaluator.batch_gen. Uses ThreadPoolExecutor (not
        multiprocessing.Pool) because LLM API calls are I/O-bound — threads
        are much more efficient than separate processes for network work.

        All items are submitted at once; results are collected via as_completed
        so the thread pool stays fully utilised at all times. A partial JSON is
        written to disk every SAVE_CHUNK_SIZE completions.
        """
        completed = []
        total = len(data)
        save_counter = 0

        def _save_partial():
            if hasattr(self, 'output_path') and self.output_path:
                partial_path = self.output_path.replace('.json', '_partial.json')
                try:
                    with open(partial_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'partial': True,
                            'completed': len(completed),
                            'total': total,
                            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'results': completed,
                        }, f, ensure_ascii=False, indent=2)
                    logger.info(f"[{self.model_name}] Partial save: {len(completed)}/{total} → {partial_path}")
                except Exception as e:
                    logger.warning(f"[{self.model_name}] Could not write partial save: {e}")

        with ThreadPoolExecutor(max_workers=self.process_num) as executor:
            pbar = tqdm(total=total, desc=f'{self.model_name}_{data_name}')
            # Submit ALL items up front so the pool is always kept busy
            future_to_item = {executor.submit(self.gen, item): item for item in data}
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    completed.append(result)
                except Exception as e:
                    logger.error(f"[{self.model_name}] Worker error: {e}")
                    completed.append(future_to_item[future])  # keep original item on failure
                pbar.update(1)
                save_counter += 1
                if save_counter % self.SAVE_CHUNK_SIZE == 0:
                    _save_partial()
            pbar.close()

        # Final partial save in case total isn't a multiple of SAVE_CHUNK_SIZE
        if save_counter % self.SAVE_CHUNK_SIZE != 0:
            _save_partial()

        return completed

    def evaluate(self, data_path: str, output_path: str, data_size: int = None) -> dict:
        """
        Override of Evaluator.evaluate that sets self.output_path BEFORE calling
        batch_gen, so partial saves inside batch_gen have a valid path to write to.
        """
        from pathlib import Path
        import datetime as _dt

        data = self.load_data(data_path, data_size)
        data_name = Path(data_path).stem
        data_size = len(data)

        info = {
            'llm': {
                "model_name": self.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "top_p": self.model.top_p
            },
            'dataset': data_name,
            'data_num': data_size,
            'datetime': _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Set output_path BEFORE batch_gen so partial saves work
        os.makedirs(output_path, exist_ok=True)
        timestamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'Eval_Judge_{self.model_name}_{data_name}_{data_size}_{timestamp}.json'
        self.output_path = os.path.join(output_path, output_name)

        self.construct_prompt(data)
        results = self.batch_gen(data, data_name)
        stat_info = self.stat_results(results)

        self.save_output({'info': info, 'stat_info': stat_info, 'results': results})

        # Clean up the _partial.json now that the full file is saved
        partial_path = self.output_path.replace('.json', '_partial.json')
        if os.path.exists(partial_path):
            os.remove(partial_path)

        return stat_info


class HallucinationEvaluator(DivergenceEvaluator):
    def __init__(self, model: Model, process_num: int = 5):
        super().__init__(model, process_num)
        self.prompt = '''You are an expert evaluator trained to assess the **hallucination level** of a model's reasoning in text-based tasks.

You will receive:
- A **question** (text only),
- A **model-generated chain of thought (output)**,
- A **ground-truth chain of thought (answer)**.

Your task is to evaluate how much the model's reasoning (**output**) includes **unsupported, fabricated, or inaccurate details** that are not grounded in the provided **answer**.  
Hallucination refers to **added entities, assumptions, facts, steps, or interpretations** that cannot be justified by the ground truth.

**Be careful not to penalize harmless rephrasings, synonymous wording, or logically equivalent reasoning.**  
Small differences in wording or structure do **not** count as hallucination if they preserve the meaning and stay faithful to the context.

---

### Evaluation Guidelines

Rate **hallucination** on a Likert scale from **1 to 5**, based primarily on the **number and severity of added or contradicted objects, facts, or reasoning steps**:

- **5** → Virtually no hallucination. The output may rephrase or paraphrase ideas, but introduces *no new factual objects* or incorrect claims.  
- **4** → Very minor hallucination. One or two mild additions appear, but they do not materially change the reasoning or introduce contradictions.  
- **3** → Moderate hallucination. A few added or altered facts or reasoning steps appear that are not supported by the ground truth.  
- **2** → High hallucination. Several important unsupported or contradictory claims are introduced.  
- **1** → Severe hallucination. The output introduces *many* fabricated objects or contradictory claims, or diverges substantially from the ground truth’s factual basis.

When rating, focus on:
- Whether **objects**, facts, or reasoning steps in the **output** appear in or are justified by the **answer**.  
- The **count** and **impact** of new, unsupported entities or claims.  
- Whether the output's additions change the meaning, introduce speculation, or contradict the answer.  
- **Do not penalize** paraphrasing, rewording, or alternative but valid restatements of the same objects/ideas.

---

### Input

**Question:**  
"""{question}"""

**Model Output (Predicted Chain of Thought):**  
"""{output}"""

**Ground Truth Chain of Thought:**  
"""{answer}"""

---

### Output Format

The **first line** of your response must contain only the JSON rating in the following format:  
{{"rating": <integer between 1 and 5>}}

Subsequent lines should provide a **brief textual justification** explaining why you assigned that rating.

Example:  
{{"rating": 2}}  
The output introduces several new entities and speculative steps not present in the ground truth, leading to a high hallucination score.
'''



def _extract_final_answer(text: str) -> str:
    """Extract the text after the first 'Final Answer:' marker (case-insensitive).
    Falls back to the full text if the marker is not found."""
    import re
    match = re.search(r'(?i)final\s+answer\s*[:\-]?\s*', text)
    if match:
        return text[match.end():].strip()
    return text.strip()


class FinalanswerEvaluator(DivergenceEvaluator):
    """Evaluates correctness of the final answer by comparing the extracted
    'Final Answer' section of the model output against the ground truth's
    'Final Answer' section, using the same Likert-scale judge framework."""

    def __init__(self, model: Model, process_num: int = 5):
        super().__init__(model, process_num)
        self.prompt = '''You are an expert evaluator assessing the **correctness of a model's final answer** relative to a ground-truth final answer for a text-based question.

You will receive:
- A **question** (text only),
- A **model final answer (output)** — extracted directly from the model's response,
- A **ground-truth final answer (answer)** — the reference correct answer.

Your task is to evaluate how correct and semantically consistent the model's final answer is compared to the ground-truth final answer.

**Important:** Focus *only* on correctness. Do not penalize for stylistic differences, verbosity, or minor paraphrasing that preserves meaning. **Be slightly lenient**: if the model’s answer captures the main idea and does not contradict the ground truth, prefer a better score rather than a worse one. If the model adds extra details beyond the ground truth, do NOT penalize it for being verbose—reward it slightly if those added details are plausible, non-contradictory, and the core answer remains accurate. Do not consider the reasoning chain.

---

### Evaluation Guidelines (Lenient, Reversed Scale)

Rate **final answer correctness** on a Likert scale from **1 to 5**:

- **5** → Perfect match in meaning with the ground truth. Minor wording differences are fine. **Any extra details are consistent (or at least not clearly wrong) and do not change the core answer.**
- **4** → Largely correct and captures the main point, with only minor omissions or slight inaccuracies. **Extra details may be somewhat speculative but not clearly contradictory.**
- **3** → Mixed: gets some key parts right but misses important elements or includes noticeable inaccuracies. **Extra details may introduce confusion or mild contradictions.**
- **2** → Mostly wrong or misses the main point, even if it shares a few surface keywords with the ground truth.
- **1** → Completely wrong, unrelated, or directly contradicts the ground truth.

When rating, focus on:
- Whether the core claim/conclusion matches the ground truth.
- Whether key entities, facts, or findings are correctly stated.
- Semantic consistency and faithfulness to the ground truth, not surface-level wording.
- **Leniency principle:** if you're unsure whether something is a minor vs. major issue, **err toward the higher score (more correct)** unless there is a clear contradiction.
- **Added details:**  
  - **Good:** extra context/examples/clarifications that remain consistent with the ground truth.  
  - **Bad:** extra details that clearly conflict with the ground truth or change the meaning.

---

### Input

**Question:**  
"""{question}"""

**Model Final Answer (output):**  
"""{output}"""

**Ground-Truth Final Answer (answer):**  
"""{answer}"""

---

### Output Format

The **first line** of your response must contain only the JSON rating in the following format:  
{{"rating": <integer between 1 and 5>}}

Subsequent lines should provide a **brief textual justification** explaining why you assigned that rating, including whether any added details helped or hurt correctness.

Example:  
{{"rating": 4}}  
The model captures the main conclusion correctly and adds extra explanation that is consistent with the ground truth, though it slightly overstates one detail.
'''

    def construct_prompt(self, data: list[dict]) -> None:
        """Extracts 'Final Answer' sections before building the judge prompt."""
        for item in data:
            model_final = _extract_final_answer(item['llm_output'])
            gt_final    = _extract_final_answer(item['correct_answer'])
            # Store extracted answers so they appear in the output JSON
            item['extracted_model_final_answer'] = model_final
            item['extracted_gt_final_answer']    = gt_final
            user_input = self.prompt.format(
                question=item['question'],
                output=model_final,
                answer=gt_final,
            )
            item['prompt'] = user_input


# Map facet names to evaluator class names (handles multi-word facet keys)
_FACET_CLASS_MAP = {
    "divergence":   "DivergenceEvaluator",
    "hallucination": "HallucinationEvaluator",
    "finalanswer":  "FinalanswerEvaluator",
}


def evaluate_model_on_facet(model_name, config, facet, data_path, output_base_dir,
                            shared_results, results_lock, csv_path, num_workers: int = 4):
    """
    Evaluates one judge model on all datasets for a given facet.
    Appends results to shared_results and flushes CSV after every file.
    """
    print(f"Starting parallel evaluation with model: {model_name} (Facet: {facet})")
    
    try:
        # Initialize model
        model = Model(
            model_name=model_name,
            model_path_or_url=config["base_url"],
            inference_mode="api",
            api_key=config["api_key"],
            max_tokens=4096,  # Allow generous output for rating JSON + justification
        )
        
        # Initialize evaluator
        class_name = _FACET_CLASS_MAP.get(facet.lower(), f"{facet.capitalize()}Evaluator")
        evaluator_class = globals()[class_name]
        evaluator = evaluator_class(model=model, process_num=num_workers)
        
        output_subdir = f"{model_name.lower()}_{facet}"
        target_data_dir = os.path.join(output_base_dir, output_subdir)
        os.makedirs(target_data_dir, exist_ok=True)
        
        # Run evaluation on all JSON files in data_path
        files = [f for f in os.listdir(data_path) if f.endswith(".json")]
        for file in files:
            full_file_path = os.path.join(data_path, file)
            print(f"[{model_name}] Processing file: {file}")
            
            try:
                results = evaluator.evaluate(
                    data_path=full_file_path,
                    # data_size=1,  # Uncomment for testing single sample
                    output_path=target_data_dir,
                )
                print(f"[{model_name}] Results for {file}: {results}")
                
                name = file.split(".")[0]
                res_entry = OrderedDict([("model_name", model_name), ("file_name", name), *results.items()])

                # --- Incremental save: flush every 100 entries ---
                with results_lock:
                    shared_results.append(res_entry)
                    if len(shared_results) % 100 == 0:
                        df = pd.DataFrame(shared_results)
                        df.to_csv(csv_path, index=False)
                        print(f"[{model_name}] Checkpoint: {len(shared_results)} entries saved to {csv_path}")

            except Exception as e:
                print(f"[{model_name}] Failed to evaluate {file}: {e}")
                logger.error(f"[{model_name}] {e}")
                
    except Exception as e:
        print(f"Critical error in evaluator setup for {model_name}: {e}")
        logger.exception(e)


# ========================
# BATCH EVALUATION
# ========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-judge CoT evaluation.")
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=64,
        help="Number of parallel workers (controls both the outer ThreadPoolExecutor "
             "for judge models AND the inner ThreadPoolExecutor for LLM calls). "
             "Default: 16. For API-bound workloads you can go much higher (32, 64).",
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="./CoT_eval_qwen_omni_only/",
        help="Path to directory containing input JSON files. Default: ./CoT_eval_qwen_omni_only/",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./CoT_eval_results_qwen_omni_only/",
        help="Base output directory for results. Default: ./CoT_eval_results_qwen_omni_only/",
    )
    parser.add_argument(
        "--facets", "-f",
        nargs="+",
        default=["finalanswer", "hallucination", "divergence"],
        choices=list(_FACET_CLASS_MAP.keys()),
        help="Facets to evaluate. Default: finalanswer. "
             "Choices: divergence hallucination finalanswer.",
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_base_dir = args.output_dir
    facets = args.facets
    num_workers = args.workers
    print(f"Workers: {num_workers} | Facets: {facets} | Data: {data_path} | Output: {output_base_dir}")
    
    os.makedirs("out", exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # --- Fully parallel: one thread per (judge_model × facet) combination ---
    # Build per-facet state up front so every thread has its own lock + results list
    facet_state: dict[str, dict] = {
        facet: {
            "shared_results": [],
            "results_lock": threading.Lock(),
            "csv_path": f"out/results_all_judges_{facet}.csv",
        }
        for facet in facets
    }

    total_combos = len(JUDGE_CONFIGS) * len(facets)
    with ThreadPoolExecutor(max_workers=min(num_workers, total_combos)) as executor:
        future_to_key = {}
        for facet in facets:
            state = facet_state[facet]
            for model_name, config in JUDGE_CONFIGS.items():
                f = executor.submit(
                    evaluate_model_on_facet,
                    model_name, config, facet, data_path, output_base_dir,
                    state["shared_results"], state["results_lock"], state["csv_path"],
                    num_workers,
                )
                future_to_key[f] = (model_name, facet)

        for future in as_completed(future_to_key):
            model_name, facet = future_to_key[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{model_name}|{facet}] raised an exception: {e}")
                logger.exception(e)

    # Final flush for every facet
    for facet, state in facet_state.items():
        shared_results = state["shared_results"]
        csv_path = state["csv_path"]
        if shared_results:
            df = pd.DataFrame(shared_results)
            df.to_csv(csv_path, index=False)
            print(f"\nAll results for {facet} saved to: {csv_path} ({len(shared_results)} entries)")

if __name__ == "__main__":
    main()
