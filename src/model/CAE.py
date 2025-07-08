from typing import Callable, Union, Any
import inspect

import torch
import torch.nn as nn


class ConditionalAutoencoder(nn.Module):
    """
    Seq-to-seq auto-encoder condizionato su vettori di proprietà chimiche.
    - Encoder: LSTM -> vettore latente deterministico
    - Decoder: LSTM condizionato (latente + proprietà)
    """

    def __init__(
        self,
        vocab_size,
        num_prop,
        latent_size=256,
        emb_size=128,
        hidden_size=512,
        n_rnn_layer=3,
        encoder_biLSTM=True,
        decoder_biLSTM=False,
    ):

        super().__init__()
        self.num_prop = num_prop
        self.encoder_biLSTM = encoder_biLSTM
        self.decoder_biLSTM = decoder_biLSTM

        # --- Embedding per token
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # --- Encoder -----------------------------------------------
        encoder_hidden_size = hidden_size
        self.encoder_rnn = nn.LSTM(
            input_size=emb_size + self.num_prop,
            hidden_size=encoder_hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=encoder_biLSTM,
        )

        # --- Decoder -------------------------------------------------
        # La dimensione dell'input per il layer latente dipende dalla bidirezionalità
        encoder_output_size = encoder_hidden_size * 2 if encoder_biLSTM else encoder_hidden_size
        self.to_latent = nn.Linear(encoder_output_size, latent_size)

        # La dimensione nascosta del decoder può essere diversa
        decoder_hidden_size = encoder_output_size  # Esempio: la stessa dell'output dell'encoder
        self.decoder_rnn = nn.LSTM(
            input_size=emb_size + latent_size + num_prop,
            hidden_size=decoder_hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=decoder_biLSTM,
        )
        self.output_linear = nn.Linear(decoder_hidden_size, vocab_size)

        # Inizializzazione dei pesi
        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializza i pesi dei layer lineari e degli embedding."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
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
        c_expanded = c.unsqueeze(1).repeat(1, T, 1)  # (B, T, num_prop)
        enc_inp = torch.cat([emb, c_expanded], dim=-1)  # (B, T, E+prop)

        packed = nn.utils.rnn.pack_padded_sequence(
            enc_inp, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.encoder_rnn(packed)  # h_n: (layers, B, H)

        # Estrai e concatena lo stato nascosto finale del forward e backward pass
        if self.encoder_rnn.bidirectional:
            # h_n ha shape (num_layers * 2, B, hidden_size)
            # Prendiamo l'ultimo layer: forward h_n[-2] e backward h_n[-1]
            h_last_layer = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        else:
            # h_n ha shape (num_layers, B, hidden_size)
            h_last_layer = h_n[-1, :, :]

        # Proietta l'output dell'encoder nello spazio latente
        z = self.to_latent(h_last_layer)  # (B, latent_size)

        # --- Decoder (teacher forcing) -----------------------------
        # concateniamo z e C a OGNI passo di tempo
        z_expanded = z.unsqueeze(1).repeat(1, T, 1)  # (B, T, latent)
        # Usiamo lo stesso `emb` e `c_expanded` perché stiamo facendo teacher forcing
        decoder_input = torch.cat([emb, z_expanded, c_expanded], dim=-1)

        dec_out, _ = self.decoder_rnn(decoder_input)  # (B, T, H)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.999)):
        device_type = self.parameters().__next__().device

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(
        #     f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        # )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # print(f"using fused AdamW: {use_fused}")

        return optimizer

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
