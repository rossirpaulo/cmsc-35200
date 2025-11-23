import json
import os
import time
import argparse
import yaml
from openai import OpenAI
import re

def load_model_config(model_shortname, config_file="model_servers.yaml"):
    if os.path.isabs(config_file):
        yaml_path = config_file
    else:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    for server in config["servers"]:
        if server["shortname"] == model_shortname:
            return {
                "api_key": server.get("openai_api_key", "no_key"),
                "api_base": server["openai_api_base"],
                "model_name": server["openai_model"]
            }
    raise ValueError(f"Model {model_shortname} not found")

def get_answer(client, model_name, prompt):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def grade_answer(client, model_name, question, model_answer, reference_answer):
    prompt = f"""
You are a strict grader.
Question: {question}
Reference Answer: {reference_answer}
Student Answer: {model_answer}

Does the Student Answer convey the same meaning as the Reference Answer?
Respond with strictly 'CORRECT' or 'INCORRECT'.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip().upper()
        if "CORRECT" in content and "INCORRECT" not in content:
            return True
        if "INCORRECT" in content:
            return False
        # Fallback heuristic
        return "CORRECT" in content
    except Exception as e:
        print(f"Grading error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_file")
    parser.add_argument("--model", required=True)
    parser.add_argument("--grader", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with open(args.questions_file, 'r') as f:
        questions = json.load(f)
    
    questions = questions[:args.limit]

    # Setup clients
    model_conf = load_model_config(args.model)
    grader_conf = load_model_config(args.grader)
    
    model_client = OpenAI(api_key=model_conf["api_key"], base_url=model_conf["api_base"])
    grader_client = OpenAI(api_key=grader_conf["api_key"], base_url=grader_conf["api_base"])

    results = []
    
    print(f"Running CoT Experiment on {len(questions)} questions...")
    print(f"Model: {args.model}, Grader: {args.grader}")

    for i, q in enumerate(questions):
        q_text = q["question"]
        ref_ans = q["answer"]
        
        print(f"\nQuestion {i+1}: {q_text[:50]}...")

        # 1. Standard Prompt
        start = time.time()
        std_ans = get_answer(model_client, model_conf["model_name"], q_text)
        std_time = time.time() - start
        std_correct = grade_answer(grader_client, grader_conf["model_name"], q_text, std_ans, ref_ans)
        
        # 2. CoT Prompt
        cot_prompt = f"{q_text}\n\nThink step by step before answering. Explain your reasoning, then provide the final answer."
        start = time.time()
        cot_ans = get_answer(model_client, model_conf["model_name"], cot_prompt)
        cot_time = time.time() - start
        cot_correct = grade_answer(grader_client, grader_conf["model_name"], q_text, cot_ans, ref_ans)

        print(f"  Standard: {'CORRECT' if std_correct else 'INCORRECT'} ({std_time:.2f}s)")
        print(f"  CoT:      {'CORRECT' if cot_correct else 'INCORRECT'} ({cot_time:.2f}s)")
        
        results.append({
            "question": q_text,
            "standard": {"answer": std_ans, "correct": std_correct, "time": std_time},
            "cot": {"answer": cot_ans, "correct": cot_correct, "time": cot_time}
        })

    # Summary
    std_acc = sum(1 for r in results if r["standard"]["correct"]) / len(results)
    cot_acc = sum(1 for r in results if r["cot"]["correct"]) / len(results)
    
    print("\nResults Summary:")
    print(f"Standard Accuracy: {std_acc*100:.1f}%")
    print(f"CoT Accuracy:      {cot_acc*100:.1f}%")
    
    # Save detailed results
    with open("cot_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

