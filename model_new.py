# model.py  (PyTorch implementation)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalAutoencoder(nn.Module):
    """
    Seq-to-seq auto-encoder condizionato su vettori di proprietà chimiche.
    - Encoder: LSTM -> vettore latente deterministico
    - Decoder: LSTM condizionato (latente + proprietà)
    """

    def __init__(self, vocab_size, args):
        """
        Args
        ----
        vocab_size : int          # dimensione vocabolario
        args.batch_size : int     # usato in sample()
        args.latent_size : int
        args.unit_size : int      # hidden size LSTM
        args.n_rnn_layer : int
        args.num_prop   : int     # n. proprietà condizionali (5 nel tuo caso)
        """
        super().__init__()

        self.vocab_size  = vocab_size
        self.latent_size = args.latent_size
        self.unit_size   = args.unit_size
        self.n_layers    = args.n_rnn_layer
        self.num_prop    = args.num_prop
        self.batch_size  = args.batch_size

        # --- Embedding per token
        self.embedding = nn.Embedding(vocab_size, args.latent_size)

        # --- Encoder -------------------------------------------------
        self.encoder_rnn = nn.LSTM(
            input_size = args.latent_size + self.num_prop,   # token emb + prop
            hidden_size = self.unit_size,
            num_layers = self.n_layers,
            batch_first = True,
            bidirectional = False
        )
        # Proiezione a vettore latente
        self.to_latent = nn.Linear(self.unit_size, self.latent_size)

        # --- Decoder -------------------------------------------------
        self.decoder_rnn = nn.LSTM(
            input_size = args.latent_size + self.latent_size + self.num_prop,
            hidden_size = self.unit_size,
            num_layers = self.n_layers,
            batch_first = True,
            bidirectional = False
        )
        self.output_linear = nn.Linear(self.unit_size, vocab_size)

        # Inizializzazione xavier per stabilità
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # ---------------------------------------------------------------
    # Forward per training (teacher forcing: X_in, Y_target)
    # ---------------------------------------------------------------
    def forward(self, x, c, lengths):
        """
        x : LongTensor  (B, T)  input sequence (con 'X' all'inizio)
        c : FloatTensor (B, num_prop)
        lengths : LongTensor (B,)  lunghezze reali (senza padding)
        """
        B, T = x.size()

        # --- Encoder ------------------------------------------------
        emb = self.embedding(x)                      # (B, T, E)
        c_exp = c.unsqueeze(1).repeat(1, T, 1)       # (B, T, num_prop)
        enc_inp = torch.cat([emb, c_exp], dim=-1)    # (B, T, E+prop)

        packed = nn.utils.rnn.pack_padded_sequence(
            enc_inp, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.encoder_rnn(packed)       # h_n: (layers, B, H)
        h_last = h_n[-1]                             # (B, H)

        z = self.to_latent(h_last)                   # (B, latent_size)

        # --- Decoder (teacher forcing) -----------------------------
        # concateniamo z e C a OGNI passo di tempo
        z_rep  = z.unsqueeze(1).repeat(1, T, 1)      # (B, T, latent)
        dec_in = torch.cat([emb, z_rep, c_exp], dim=-1)

        dec_out, _ = self.decoder_rnn(dec_in)        # (B, T, H)
        logits = self.output_linear(dec_out)         # (B, T, vocab)
        return z, logits, dec_out                    # z restituito per debug

    # ---------------------------------------------------------------
    # Sampling autoregressivo (greedy o multinomial)
    # ---------------------------------------------------------------
    def sample(self, latent_vector, c, start_token, seq_length=120,
               greedy=False, device="cpu"):
        """
        latent_vector : FloatTensor (B, latent_size)
        c             : FloatTensor (B, num_prop)
        start_token   : LongTensor  (B, 1)   token 'X'
        Returns: LongTensor (B, T_generated)
        """
        B = latent_vector.size(0)
        x = start_token.to(device)
        hidden = None
        generated = []

        for _ in range(seq_length):
            x_emb  = self.embedding(x)                    # (B, 1, E)
            z_exp  = latent_vector.unsqueeze(1)           # (B, 1, latent)
            c_exp  = c.unsqueeze(1)                       # (B, 1, prop)
            dec_in = torch.cat([x_emb, z_exp, c_exp], -1) # (B, 1, cat)

            out, hidden = self.decoder_rnn(dec_in, hidden)   # (B, 1, H)
            logits = self.output_linear(out.squeeze(1))      # (B, vocab)

            if greedy:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # (B,1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)                 # (B,1)

            generated.append(next_idx)
            x = next_idx

        return torch.cat(generated, dim=1)  # (B, seq_len)

    # ---------------------------------------------------------------
    # Utility per salvare/caricare pesi PyTorch
    # ---------------------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
