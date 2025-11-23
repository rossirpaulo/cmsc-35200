import json
import glob
import os
from datetime import datetime

def get_latest_result_files(directory):
    files = glob.glob(os.path.join(directory, "results_*.json"))
    
    # Dictionary to store latest file for each (grader, tester) pair
    latest_files = {}
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                
            meta = data.get('metadata', {})
            grader = meta.get('grader_model')
            tester = meta.get('test_model')
            timestamp_str = meta.get('timestamp')
            
            if not (grader and tester and timestamp_str):
                continue
                
            # Normalize model names if needed (e.g., if they differ slightly)
            
            key = (grader, tester)
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                continue
                
            if key not in latest_files or timestamp > latest_files[key]['timestamp']:
                latest_files[key] = {
                    'file': f,
                    'timestamp': timestamp,
                    'data': data
                }
        except Exception as e:
            continue
            
    return latest_files

def main():
    directory = "lab3/argonium"
    latest_results = get_latest_result_files(directory)
    
    # Define the models order for consistent output
    models = ["llama70", "oss120", "gemma2", "llama3.2"]
    
    print(f"{'='*80}")
    print(f"CROSS-GRADING DETAILS FOR FINAL QUESTION (Q30)")
    print(f"{'='*80}\n")
    
    for grader in models:
        for tester in models:
            key = (grader, tester)
            if key not in latest_results:
                print(f"MISSING RESULT: Grader={grader}, Tester={tester}")
                continue
                
            result_data = latest_results[key]['data']
            results_list = result_data.get('results', [])
            
            if not results_list:
                print(f"NO RESULTS FOUND: Grader={grader}, Tester={tester}")
                continue
                
            # Get the last question
            last_q = results_list[-1]
            q_id = last_q.get('question_id')
            
            print(f"🔹 GRADER: {grader} | 🔸 TESTER: {tester} | Q{q_id}")
            print(f"{'-'*40}")
            
            # Model Answer (truncated if too long)
            ans = last_q.get('model_answer', 'N/A').replace('\n', ' ')
            if len(ans) > 150:
                ans = ans[:147] + "..."
            print(f"🗣️  MODEL ANSWER: {ans}")
            
            # Evaluation
            eval_data = last_q.get('evaluation', {})
            score = eval_data.get('score')
            reasoning = eval_data.get('reasoning', 'No reasoning provided')
            if isinstance(reasoning, str) and len(reasoning) > 150:
                reasoning = reasoning[:147] + "..."
                
            status = "✅ CORRECT" if score == 1 else "❌ INCORRECT"
            print(f"📝 GRADER EVAL: {status} (Score: {score})")
            print(f"🤔 REASONING:   {reasoning}")
            print(f"\n")

if __name__ == "__main__":
    main()

