import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer, XLNetTokenizerFast

import string
import multiprocessing


from neural_tokenizer import LSTMNeuralTokenizer, tag_vocab


def get_vocab(langs="en"):
    langs = langs.split(",")
    vocab = dict()
    vocab["<UNK>"] = len(vocab)
    vocab["<PAD>"] = len(vocab)
    vocab["<SEP>"] = len(vocab)
    vocab["<CLS>"] = len(vocab)
    vocab["<MASK>"] = len(vocab)

    letters = string.ascii_letters
    digits = string.digits
    punctuation = string.punctuation
    space = " "
    en_chars = list(letters + digits + punctuation + space)
    vocab.update({char: idx for idx, char in enumerate(en_chars, start=len(vocab))})

    if "es" in langs:
        spanish_letters = "áéíóúüñÁÉÍÓÚÜÑ"
        spanish_punctuation = "¡¿"
        spanish_chars = list(spanish_letters + spanish_punctuation)
        vocab.update({char: idx for idx, char in enumerate(spanish_chars, start=len(vocab))})

    return vocab


def generate_char_iob_tags(examples, tokenizer):
    subword_indices = tokenizer(examples, return_offsets_mapping=True)["offset_mapping"]

    iob_tags = []
    for idx, example in enumerate(examples):
        char_tags = ["O"] * len(example)
        for (start, end) in subword_indices[idx]:
            if start >= len(example):
                break
            char_tags[start] = "B"
            for i in range(start + 1, min(end, len(example))):
                char_tags[i] = "I"
        iob_tags += [char_tags]

    return iob_tags


def get_iob_data(examples, tokenizer, vocab, lang=None):
    prepared_data = {}

    iob_tags = generate_char_iob_tags(examples["premise"], tokenizer)

    prepared_data["iob_tags"] = []

    prepared_data["input_ids"] = []
    for idx, text in enumerate(examples["premise"]):
        # text = example
        if text == '':
            continue
        char_ids = [vocab[lang]] if lang else []
        char_ids += [vocab[char] if char in vocab else vocab["<UNK>"] for char in text]
        prepared_data["input_ids"] += [char_ids]

        tag_ids = [tag_vocab["O"]] if lang else []
        tag_ids += [tag_vocab[tag] for tag in iob_tags[idx]]
        prepared_data["iob_tags"] += [tag_ids]

    return prepared_data


class IOBDataset(Dataset):
    def __init__(self, char_ids, labels):
        self.char_ids = char_ids
        self.iob_tags = labels

    def __len__(self):
        return len(self.char_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.char_ids[idx], dtype=torch.long), torch.tensor(self.iob_tags[idx], dtype=torch.long)



def collate_fn(batch):
    chars, labels = zip(*batch)
    # Pad sequences to the same length
    chars_padded = pad_sequence([torch.tensor(c) for c in chars], batch_first=True, padding_value=vocab["<PAD>"])
    labels_padded = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=0)
    return chars_padded, labels_padded


def create_dataloader(char_ids, labels, batch_size=32):
    dataset = IOBDataset(char_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def train_tokenizer(model, dataloader, num_epochs=10, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        for chars, labels in tqdm(dataloader, desc=f"Epoch {epoch}: "):
            # chars, labels = batch
            chars, labels = chars.to(device), labels.to(device)
            outputs = model(chars)
            outputs = outputs.view(-1, 3)
            labels = labels.view(-1)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Tokenizer Trainer')
    parser.add_argument("--epochs", type=int, default=10, help="num_epochs")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = "cpu"
    num_proc = multiprocessing.cpu_count()

    vocab = get_vocab("en,es")

    vocab["<en>"] = len(vocab)
    vocab["<es>"] = len(vocab)
    print(len(vocab))

    en_tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
    en_dataset = load_dataset("xnli", "en", split="train")
    en_iob_dataset = (
        en_dataset.remove_columns([col for col in en_dataset.column_names if col != "premise"])
        .map(get_iob_data, batched=True, num_proc=num_proc,
             fn_kwargs={"tokenizer": en_tokenizer, "vocab": vocab, "lang": "<en>"})
    )

    es_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    es_dataset = load_dataset("xnli", "es", split="train")
    es_iob_dataset = (
        es_dataset.remove_columns([col for col in es_dataset.column_names if col != "premise"])
        .map(get_iob_data, batched=True, num_proc=num_proc,
             fn_kwargs={"tokenizer": es_tokenizer, "vocab": vocab, "lang": "<es>"})
    )

    input_ids = en_iob_dataset["input_ids"] + es_iob_dataset["input_ids"]
    iob_tags = en_iob_dataset["iob_tags"] + es_iob_dataset["iob_tags"]

    model = LSTMNeuralTokenizer(vocab_size=len(vocab), char_emb_dim=768)
    dataloader = create_dataloader(input_ids, iob_tags, batch_size=64)
    train_tokenizer(model, dataloader, num_epochs=args.epochs)

    torch.save(model.state_dict(), "en_es_neural_tokenizer.pth")

    ## Testing
    print("Tesing tokenizer: ")
    text = "I am testing a tokenizer"
    labels = generate_char_iob_tags([text], en_tokenizer)[0]
    tag_ids = [tag_vocab[tag] for tag in labels]
    char_ids = [vocab["<en>"]] + [vocab[char] if char in vocab else vocab["<UNK>"] for char in text]
    char_ids = torch.tensor(char_ids, dtype=torch.long).to(device)

    output = model(char_ids)
    print("Expected tags:", tag_ids)
    print("Predicted tags:", torch.argmax(output, dim=1))