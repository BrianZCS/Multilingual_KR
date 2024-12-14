from datasets import load_dataset
import random

# Load the dataset
dataset = load_dataset("zares/wiki.tr.txt")
data_column = dataset["train"]["text"]  # Adjust the split if needed

# Define output file sizes
output_sizes = [1000, 5000, 10000]
output_files = ["wiki_tr_1000.txt", "wiki_tr_5000.txt", "wiki_tr_10000.txt"]

# Write subsets to files
start_index = 0
for size, output_file in zip(output_sizes, output_files):
    with open(output_file, "w") as file:
        subset = data_column[start_index:start_index + size]
        for line in subset:
            file.write(line.strip() + "\n")  # Remove newlines and write
    print(f"File saved to: {output_file}")