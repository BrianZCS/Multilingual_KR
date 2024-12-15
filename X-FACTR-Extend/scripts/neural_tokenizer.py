from torch import nn
import torch

tag_vocab = {"O": 0, "I": 1, "B": 2}


class LSTMNeuralTokenizer(nn.Module):
    def __init__(self, vocab_size, char_emb_dim=64, lstm_dim=128):
        super(LSTMNeuralTokenizer, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_emb_dim)
        self.lstm = nn.LSTM(char_emb_dim, lstm_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_dim * 2, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids):
        x = self.char_embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        # return logits
        return self.softmax(logits)

    def pool_embeddings(self, labels, char_input_ids, device):
        """
        Pool character-level embeddings into token-level embeddings.
        """
        batch_size, seq_len = labels.size()
        pooled_embeddings = []
        for batch in range(batch_size):
            token_embeds = []
            current_token = []
            for i in range(seq_len):
                label = labels[batch, i]
                char_id = char_input_ids[batch][i]
                if label in [2, 0] and current_token != []:
                    token_embeds.append(torch.mean(self.char_embedding(torch.tensor(current_token).to(device)), dim=0))
                    current_token = []
                if label != 0:
                    current_token.append(char_id)
            pooled_embeddings.append(torch.stack(token_embeds))
        return torch.nn.utils.rnn.pad_sequence(pooled_embeddings, batch_first=True)
