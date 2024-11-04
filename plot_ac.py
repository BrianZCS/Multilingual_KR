import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace these with your actual data)
languages = ['en', 'es', 'zh', 'tr']
independent_mbert = [08.397034225369258, 9.102240315633696, 3.777406835729453, 0.9893944342236416]  # M-BERT independent scores for en, zh, tr
independent_bert = [0.019580450015230664, 0.04281512506654106, 0, 0]     #  independent scores for en, zh, tr
# confidence_mbert = [10, 9, 2]    # M-BERT confidence-based scores for en, zh, tr
# confidence_bert= [6, 3, 1]      # confidence-based scores for en, zh, tr

# Plot settings
width = 0.18  # Bar width (smaller to increase space between bars)
gap = 0.05    # Gap between M-BERT and XLM for each language
x = np.arange(len(languages))  # X-axis positions

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Independent bars (solid)
#for i, mbert  in enumerate(independent_mbert):
for i, (mbert, bert) in enumerate(zip(independent_mbert, independent_bert)):
    ax.bar(x[i] - width - gap / 2, mbert, width=width, label='Independent: M-BERT tokenizer' if i == 0 else "", color='blue', edgecolor='blue')
    ax.bar(x[i] + gap / 2, bert, width=width, label='Independent: Bert language-specific tokenizer' if i == 0 else "", color='orange', edgecolor='orange',)

# Confidence-based bars (empty but with color for smaller bars)
# for i, (conf_mbert, conf_bert) in enumerate(zip(confidence_mbert, confidence_bert)):
#     mbert_diff = independent_mbert[i] - conf_mbert
#     xlm_diff = independent_xlm[i] - conf_bert

#     if mbert_diff <= 0:
#         ax.bar(x[i] - width - gap / 2, mbert_diff, width=width, bottom=conf_mbert, edgecolor='blue', facecolor='white', linewidth=1.5,
#             label='Confidence-based: M-BERT' if i == 0 else "")
#     else:
#         ax.bar(x[i] - width - gap / 2, conf_mbert, width=width, edgecolor='blue', facecolor='white', linewidth=1.5,
#                label='Confidence-based: M-BERT' if i == 0 else "")
#     if xlm_diff <= 0:
#         ax.bar(x[i] + gap / 2, xlm_diff, width=width, bottom=conf_xlm, edgecolor='orange', facecolor='white', linewidth=1.5,
#             label='Confidence-based: Bert' if i == 0 else "")
#     else:
#         ax.bar(x[i] + gap / 2, conf_xlm, width=width, edgecolor='orange', facecolor='white', linewidth=1.5,
#                label='Confidence-based: Bert' if i == 0 else "")

# Set labels and title
ax.set_xticks(x)
ax.set_xticklabels(languages, fontsize=12)
ax.set_ylim(0, max(independent_mbert) + 1)
ax.set_xlabel('Languages')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparison of M-BERT with M-BERT tokenizer and language specific tokenizer across Languages')
ax.legend()

# Add grid and display
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("scores_mbert.png", dpi=300)
plt.show()