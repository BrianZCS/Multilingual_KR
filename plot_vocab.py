import matplotlib.pyplot as plt

# Data for the vocabulary sizes
models = ["bert-base-multilingual", "bert-base-english", "bert-base-spanish", "bert-base-chinese", "bert-base-turkish"]
vocab_sizes = [119547, 28996, 32005, 21128, 32000]

# Plotting the data with blue bars and saving the figure
plt.figure(figsize=(10, 6))
plt.bar(models, vocab_sizes, color='blue')
plt.title("Vocabulary Size of Different BERT Tokenizer")
plt.xlabel("Model/Tokenizer", size=12)
plt.ylabel("Vocabulary Size", size=12)
plt.tight_layout()

# Save the plot to a file
plt.savefig('vocabulary_size_comparison.png')  # Save the plot as a PNG file
plt.show()