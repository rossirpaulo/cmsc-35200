import json
import argparse
import os
import yaml
from openai import OpenAI

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces_file")
    parser.add_argument("--validator", required=True)
    parser.add_argument("--config", default="model_servers.yaml")
    args = parser.parse_args()

    with open(args.traces_file, 'r') as f:
        traces = json.load(f)

    val_conf = load_model_config(args.validator, args.config)
    client = OpenAI(api_key=val_conf["api_key"], base_url=val_conf["api_base"])
    model = val_conf["model_name"]

    print(f"Validating {len(traces)} traces using {args.validator}...")

    for i, item in enumerate(traces):
        q_text = item.get("question", "")
        reasoning = item.get("reasoning", {}).get("prediction", {}).get("prediction_reasoning", "")
        predicted = item.get("reasoning", {}).get("prediction", {}).get("predicted_answer", "")
        
        if not reasoning:
            print(f"Skipping trace {i+1}: No reasoning found.")
            continue

        prompt = f"""
You are a logic validator.
Question: {q_text}
Reasoning provided: {reasoning}
Predicted Answer: {predicted}

Task:
1. Analyze if the reasoning is sound and logical based on the question.
2. Verify if the reasoning logically leads to the predicted answer.
3. Respond with a JSON object: {{"valid": true/false, "critique": "explanation"}}
"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            print(f"\nTrace {i+1} Validation:")
            print(content)
        except Exception as e:
            print(f"Error validating trace {i+1}: {e}")

if __name__ == "__main__":
    main()

