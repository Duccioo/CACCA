# model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConditionalAutoencoder(nn.Module):
    def __init__(self, vocab_size, args):
        super(ConditionalAutoencoder, self).__init__()

        self.vocab_size = vocab_size
        self.latent_size = args.latent_size
        self.num_prop = args.num_prop
        self.unit_size = args.unit_size
        self.n_rnn_layer = args.n_rnn_layer
        self.batch_size = args.batch_size

        # Encoder
        self.embedding_encode = nn.Embedding(self.vocab_size, self.latent_size)
        self.encoder_rnn = nn.LSTM(
            input_size=self.latent_size + self.num_prop,
            hidden_size=self.unit_size,
            num_layers=self.n_rnn_layer,
            batch_first=True,
        )
        self.latent_linear = nn.Linear(self.unit_size, self.latent_size)

        # Decoder
        self.embedding_decode = nn.Embedding(self.vocab_size, self.latent_size)
        self.decoder_rnn = nn.LSTM(
            input_size=self.latent_size + self.latent_size + self.num_prop,  # Z + X + C
            hidden_size=self.unit_size,
            num_layers=self.n_rnn_layer,
            batch_first=True,
        )
        self.output_linear = nn.Linear(self.unit_size, self.vocab_size)

        print("Network Ready")

    def encode(self, x, c, l):
        x_embed = self.embedding_encode(x)
        c_expand = c.unsqueeze(1).repeat(1, x.size(1), 1)
        inp = torch.cat([x_embed, c_expand], dim=-1)

        # Packing is not strictly necessary if lengths are all the same, but good practice
        packed_input = pack_padded_sequence(
            inp, l.cpu().numpy(), batch_first=True, enforce_sorted=False
        )

        _, (h, _) = self.encoder_rnn(packed_input)

        # Use the hidden state of the last layer
        h_last = h[-1]
        latent_vector = self.latent_linear(h_last)
        return latent_vector

    def decode(self, z, c, x, l):
        z_expand = z.unsqueeze(1).repeat(1, x.size(1), 1)
        c_expand = c.unsqueeze(1).repeat(1, x.size(1), 1)
        x_embed = self.embedding_decode(x)

        inp = torch.cat([z_expand, x_embed, c_expand], dim=-1)

        packed_input = pack_padded_sequence(
            inp, l.cpu().numpy(), batch_first=True, enforce_sorted=False
        )

        output_packed, _ = self.decoder_rnn(packed_input)

        # Pass total_length to ensure output padding is consistent with target y
        output_padded, _ = pad_packed_sequence(
            output_packed, batch_first=True, total_length=x.size(1)
        )

        logits = self.output_linear(output_padded)
        reconstruction = F.softmax(logits, dim=-1)

        return reconstruction, logits

    def forward(self, x, c, l):
        latent_vector = self.encode(x, c, l)
        reconstruction, logits = self.decode(latent_vector, c, x, l)
        return reconstruction, logits, latent_vector

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_latent_vector(self, x, c, l):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.LongTensor(x)
            c_tensor = torch.FloatTensor(c)
            l_tensor = torch.LongTensor(l)
            latent_vector = self.encode(x_tensor, c_tensor, l_tensor)
        return latent_vector.cpu().numpy()

    def sample(self, latent_vector, c, start_codon, seq_length):
        self.eval()
        with torch.no_grad():
            latent_vector = torch.FloatTensor(latent_vector)
            c = torch.FloatTensor(c)

            # The initial hidden state for the decoder LSTM
            h = latent_vector.unsqueeze(0).repeat(self.n_rnn_layer, 1, 1)
            cell = torch.zeros_like(h)  # Initial cell state
            hidden = (h, cell)

            current_token = torch.LongTensor(start_codon)
            preds = []

            for _ in range(seq_length):
                x_embed = self.embedding_decode(current_token)
                z_expand = latent_vector.unsqueeze(1)
                c_expand = c.unsqueeze(1)

                inp = torch.cat([z_expand, x_embed, c_expand], dim=-1)

                output, hidden = self.decoder_rnn(inp, hidden)

                logits = self.output_linear(output.squeeze(1))

                # Sample from the output distribution
                probs = F.softmax(logits, dim=-1)
                current_token = torch.multinomial(probs, 1)
                preds.append(current_token.cpu().numpy())

        return np.concatenate(preds, axis=1).squeeze()
