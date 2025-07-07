import torch
import torch.nn as nn


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
        args.latent_size : int
        args.unit_size : int      # hidden size LSTM
        args.n_rnn_layer : int
        args.num_prop   : int     # n. proprietà condizionali (5 nel tuo caso)
        """
        super().__init__()

        self.emb_size = args.emb_size
        self.vocab_size = vocab_size
        self.latent_size = args.latent_size
        self.unit_size = args.unit_size
        self.n_layers = args.n_rnn_layer
        self.num_prop = args.num_prop
        self.biLSTM = True  # bidirezionale

        # --- Embedding per token
        self.embedding = nn.Embedding(vocab_size, args.emb_size)

        # --- Encoder -------------------------------------------------
        self.encoder_rnn = nn.LSTM(
            input_size=args.emb_size + self.num_prop,  # token emb + prop
            hidden_size=self.unit_size,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=False if not self.biLSTM else True,
        )
        # Proiezione a vettore latente
        if self.biLSTM:
            self.unit_size = 2 * self.unit_size  # bidirezionale
        self.to_latent = nn.Linear(self.unit_size, self.latent_size)

        # --- Decoder -------------------------------------------------
        self.decoder_rnn = nn.LSTM(
            input_size=args.emb_size + self.latent_size + self.num_prop,
            hidden_size=self.unit_size,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=False if not self.biLSTM else True,
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
        emb = self.embedding(x)  # (B, T, E)
        c_exp = c.unsqueeze(1).repeat(1, T, 1)  # (B, T, num_prop)
        enc_inp = torch.cat([emb, c_exp], dim=-1)  # (B, T, E+prop)

        packed = nn.utils.rnn.pack_padded_sequence(
            enc_inp, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.encoder_rnn(packed)  # h_n: (layers, B, H)
        h_last = h_n[-1]  # (B, H)

        if self.biLSTM:
            h_in = torch.cat([h_last, h_n[-2]], dim=-1)
        else:
            h_in = h_last
        z = self.to_latent(h_in)  # (B, latent_size)

        # --- Decoder (teacher forcing) -----------------------------
        # concateniamo z e C a OGNI passo di tempo
        z_rep = z.unsqueeze(1).repeat(1, T, 1)  # (B, T, latent)
        dec_in = torch.cat([emb, z_rep, c_exp], dim=-1)

        dec_out, _ = self.decoder_rnn(dec_in)  # (B, T, H)
        logits = self.output_linear(dec_out)  # (B, T, vocab)
        return z, logits, dec_out  # z restituito per debug

    # ---------------------------------------------------------------
    # Sampling autoregressivo (greedy o multinomial)
    # ---------------------------------------------------------------
    def sample_old(self, latent_vector, c, start_token, seq_length=120, greedy=False, device="cpu"):
        """
        latent_vector : FloatTensor (B, latent_size)
        c             : FloatTensor (B, num_prop)
        start_token   : LongTensor  (B, 1)   token 'X'
        Returns: LongTensor (B, T_generated)
        """
        # B = latent_vector.size(0)
        x = start_token.to(device)
        hidden = None
        generated = []

        for _ in range(seq_length):
            x_emb = self.embedding(x)  # (B, 1, E)
            z_exp = latent_vector.unsqueeze(1)  # (B, 1, latent)
            c_exp = c.unsqueeze(1)  # (B, 1, prop)
            dec_in = torch.cat([x_emb, z_exp, c_exp], -1)  # (B, 1, cat)

            out, hidden = self.decoder_rnn(dec_in, hidden)  # (B, 1, H)
            logits = self.output_linear(out.squeeze(1))  # (B, vocab)

            if greedy:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)  # (B,1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)  # (B,1)

            generated.append(next_idx)
            x = next_idx

        return torch.cat(generated, dim=1)  # (B, seq_len)

    def sample(
        self,
        latent_vector,
        c,
        start_token,
        seq_length=120,
        eos_idx=None,  # ← id di 'E'
        greedy=False,
        top_k=None,  # es. 30
        top_p=None,
        pad_idx=None,
        temperature=1.0,
        device="cpu",
    ):
        """
        Ritorna LongTensor (B, T) con token generati fino a E o padding finale.
        """
        B = latent_vector.size(0)
        x = start_token.to(device)  # (B, 1)
        hidden = None
        generated = []
        ended = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(seq_length):
            # --- passo di decodifica
            x_emb = self.embedding(x)  # (B, 1, E)
            z_exp = latent_vector.unsqueeze(1)  # (B, 1, latent)
            c_exp = c.unsqueeze(1)  # (B, 1, prop)
            dec_in = torch.cat([x_emb, z_exp, c_exp], -1)  # (B, 1, cat)

            out, hidden = self.decoder_rnn(dec_in, hidden)
            logits = self.output_linear(out.squeeze(1))  # (B, vocab)

            # --- sampling
            logits = logits / temperature  # T scaling
            if greedy:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if top_k is not None:  # top-k filtering
                    topk = torch.topk(logits, top_k)
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(1, topk.indices, topk.values)
                    logits = mask
                if top_p is not None:  # nucleus filtering
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    mask = cumprobs > top_p
                    mask[..., 0] = False  # sempre tieni il best
                    sorted_logits[mask] = float("-inf")
                    logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)  # (B,1)

            # --- handle end-of-sequence
            next_idx[ended] = pad_idx if pad_idx is not None else eos_idx
            ended |= next_idx.squeeze(1) == eos_idx
            generated.append(next_idx)
            x = next_idx
            if ended.all():
                break

        # concatena in (B, T) e, se vuoi, riempi con pad fino a seq_length
        out = torch.cat(generated, dim=1)
        # if out.size(1) < seq_length and pad_idx is not None:
        #     pad_fill = pad_idx * torch.ones(B, seq_length - out.size(1),
        #                                     dtype=torch.long, device=device)
        #     out = torch.cat([out, pad_fill], dim=1)
        return out

    # ---------------------------------------------------------------
    # Utility per salvare/caricare pesi PyTorch
    # ---------------------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
