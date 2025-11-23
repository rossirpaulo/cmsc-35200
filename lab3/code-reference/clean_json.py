import json
import re
import sys

def clean_questions(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    cleaned_count = 0
    for item in data:
        question = item.get('question', '')
        # Remove (*) marker
        if '(*)' in question:
            item['question'] = question.replace('(*)', '')
            cleaned_count += 1
            
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Cleaned {cleaned_count} questions. Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_json.py <input_json> <output_json>")
        sys.exit(1)
    clean_questions(sys.argv[1], sys.argv[2])

