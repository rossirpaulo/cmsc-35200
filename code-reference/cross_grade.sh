#!/bin/bash
# Get absolute path for venv
VENV_PATH="$PWD/lab3/.venv/bin"
export PATH=$VENV_PATH:$PATH

# Absolute path for CLEANED questions
QUESTIONS="$PWD/lab3/mcqa_output/llama70_clean.json"

# Create output dir (absolute path)
mkdir -p "$PWD/lab3/cross_grade_results"
OUT_DIR="$PWD/lab3/cross_grade_results"

# Change directory to where the scripts are
cd lab3/argonium

# List of models to test (Remote + Local)
MODELS=("llama70" "oss120" "gemma2" "llama3.2")

for grader in "${MODELS[@]}"; do
    echo "Running cross-grading with grader: $grader"
    # Run sequentially to avoid overloading local Ollama
    python run_all_models.py "$QUESTIONS" --grader "$grader" --random 30 --parallel 1 --seed 42 > "$OUT_DIR/grade_by_$grader.txt" 2>&1
done
