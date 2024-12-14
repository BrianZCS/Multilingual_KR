#!/bin/bash

# Define languages and file lengths
languages=("tr" "zh")
lengths=(1000 5000 10000)

# Loop through each combination of language and length
for lang in "${languages[@]}"; do
  # Determine the model based on the language
  if [[ "$lang" == "zh" ]]; then
    model="bert-base-chinese"
  elif [[ "$lang" == "tr" ]]; then
    model="dbmdz/bert-base-turkish-cased"
  else
    echo "Unsupported language: $lang"
    exit 1
  fi

  for length in "${lengths[@]}"; do
    input_file="wiki_${lang}_${length}.txt"
    output_dir="wiki_${lang}_${length}_output"
    epochs=5

    # Run the script and wait for it to complete
    echo "Running: run_lm_focus.sh $input_file $output_dir $epochs $model"
    ./run_lm_focus.sh "$input_file" "$output_dir" "$epochs" "$model" &
    wait  # Wait for the current background process to complete
  done
done

echo "All tasks completed."