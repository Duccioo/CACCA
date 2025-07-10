import inspect
import torch
import torch.nn as nn





class CVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_prop,
        latent_size=256,
        emb_size=128,
        hidden_size=512,
        n_rnn_layer=3,
        encoder_biLSTM=True,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.latent_size = latent_size
        
        # --- 1. Embedding per token ---
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # --- 2. Encoder (BiLSTM con condizionamento PRE-LSTM) ---
        encoder_input_dim = emb_size + num_prop
        
        self.encoder_rnn = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=encoder_biLSTM,
            dropout=dropout_rate if n_rnn_layer > 1 else 0,
        )
        
        encoder_output_size = hidden_size * (2 if encoder_biLSTM else 1)
        
        # --- 3. Proiezione allo Spazio Latente (MODIFICA CHIAVE: da CAE a CVAE) ---
        # Invece di un unico proiettore a 'z', ne creiamo due: uno per la media (mu)
        # e uno per il logaritmo della varianza (log_sigma).
        
        # Il MLP intermedio rimane lo stesso
        self.encoder_to_hidden = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.GELU(), 
        )
        
        # Due "teste" separate che partono dall'output del MLP intermedio
        self.to_mean = nn.Linear(hidden_size, latent_size)
        self.to_log_sigma = nn.Linear(hidden_size, latent_size)
        
        # --- 4. Decoder (rimane invariato) ---
        self.latent_to_decoder_hidden = nn.Linear(latent_size, hidden_size * n_rnn_layer)

        self.decoder_rnn = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            dropout=dropout_rate if n_rnn_layer > 1 else 0,
        )
        self.output_to_vocab = nn.Linear(hidden_size, vocab_size)

       # --- 5. Predittore di Proprietà (rimane invariato) ---
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_prop),
        )

        self._initialize_weights()

    def encode(self, x: torch.Tensor, c: torch.Tensor, lengths: torch.Tensor):
        """
        MODIFICATO: Ora restituisce la media e il log-sigma della distribuzione latente.
        """
        B, T_in = x.size()
        
        embedded_smiles = self.embedding(x)
        c_expanded = c.unsqueeze(1).repeat(1, T_in, 1)
        encoder_input = torch.cat([embedded_smiles, c_expanded], dim=2)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(encoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder_rnn(packed_input)
        
        h_last_layer = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)

        # 5. Proietta nello spazio dei parametri della distribuzione
        hidden = self.encoder_to_hidden(h_last_layer)
        mean = self.to_mean(hidden)
        log_sigma = self.to_log_sigma(hidden)  
        
        self.mean = mean
        self.log_sigma = log_sigma    
        
        return mean, log_sigma

    # NUOVO METODO: Reparameterization Trick
    def reparameterize(self, mean: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """
        Esegue il reparameterization trick per campionare da N(mean, sigma).
        """
        std = torch.exp(0.5 * log_sigma)
        # Campiona da una distribuzione normale standard
        epsilon = torch.randn_like(std)
        # Applica la trasformazione per ottenere il campione dalla nostra distribuzione
        return mean + epsilon * std

    def decode(self, z: torch.Tensor, x_target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Questo metodo rimane invariato. Prende in input un campione 'z' dallo spazio latente.
        """
        B, T_out = x_target.size()
        
        hidden_flat = self.latent_to_decoder_hidden(z)
        h_0 = hidden_flat.view(self.decoder_rnn.num_layers, B, self.decoder_rnn.hidden_size)
        c_0 = torch.zeros_like(h_0)
        
        embedded_target = self.embedding(x_target)
        
        packed_target = nn.utils.rnn.pack_padded_sequence(embedded_target, lengths.cpu(), batch_first=True, enforce_sorted=False)
        dec_out_packed, _ = self.decoder_rnn(packed_target, (h_0, c_0))
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(dec_out_packed, batch_first=True, total_length=T_out)

        logits = self.output_to_vocab(dec_out)
        return logits
        
    def _initialize_weights(self):
        """Inizializza i pesi dei layer lineari e degli embedding."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, lengths: torch.Tensor):
        """
        MODIFICATO: Esegue il forward pass completo del CVAE.
        Restituisce anche i parametri della distribuzione latente per il calcolo della loss.
        """
        # 1. Ottieni i parametri della distribuzione latente
        mean, log_sigma = self.encode(x, c, lengths)
        
        self.mean = mean
        self.log_sigma = log_sigma
        
        # 2. Campiona un vettore latente 'z' usando il reparameterization trick
        z = self.reparameterize(mean, log_sigma)
        
        # 3. Esegui i due task (ricostruzione e predizione) a partire da 'z'
        reconstruction_logits = self.decode(z, x, lengths)
        predicted_properties = self.property_predictor(z)
        
        # Restituiamo tutto il necessario per calcolare le loss nel loop di training
        return z, reconstruction_logits, predicted_properties
    
    def configure_optimizers(self, learning_rate, weight_decay=0.01, betas=(0.9, 0.999)):
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
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer


    def kl_loss(self, mean=None, log_std=None) -> torch.Tensor:
        """
        Computes the Kullback–Leibler (KL) divergence for a diagonal Gaussian
        with parameters (mean, log_std) relative to a standard normal distribution N(0, I).

        Given:
            log_std = log(σ),
        the KL divergence can be expressed as:

            KL = -0.5 * mean_over_batch(
                sum_over_latent_dim(1 + 2*log_std - mean^2 - exp(log_std)^2)
            )

        :param mean: A tensor of shape [batch_size, latent_dim], representing the Gaussian means.
        :param log_std: A tensor of the same shape, representing the log of the standard deviation.
        :return: A scalar tensor with the KL divergence averaged over the batch.
        """
        
        if mean is None and log_std is None:
            mean = self.mean
            log_std = self.log_sigma
        
        if mean is None or log_std is None:
            return torch.tensor(0.0, device=mean.device)

        return -0.5 * torch.mean(
            torch.sum(1 + 2 * log_std - mean**2 - log_std.exp() ** 2, dim=1)
        )
    
    
    # ---------------------------------------------------------------
    # Utility per salvare/caricare pesi PyTorch
    # ---------------------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))




class ConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_prop,
        latent_size=256,
        emb_size=128,
        hidden_size=512,
        n_rnn_layer=3,
        encoder_biLSTM=True,
        dropout_rate=0.1,
        decode_conditioning=True,  # Se True, il decoder usa 'c' come condizione
    ):
        super().__init__()
        self.num_prop = num_prop
        self.latent_size = latent_size
        self.decode_conditioning = decode_conditioning

        # --- Embedding per token
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # --- 2. Encoder (BiLSTM con condizionamento PRE-LSTM) ---
        # L'input della nostra LSTM sarà la concatenazione dell'embedding del token
        # e del vettore di condizione.
        encoder_input_dim = emb_size + num_prop
        
        self.encoder_rnn = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=encoder_biLSTM, # Usiamo BiLSTM per un contesto completo
            dropout=dropout_rate if n_rnn_layer > 1 else 0,
        )
        
        encoder_output_size = hidden_size * (2 if encoder_biLSTM else 1)
        
        
        # --- 3. Proiezione allo Spazio Latente ---
        # Un semplice MLP per mappare l'output dell'encoder a z.
        # Questo output è già condizionato, quindi non dobbiamo più aggiungere 'c'.
        self.to_latent = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.GELU(), 
            nn.Linear(hidden_size, latent_size)
        )
        
        # --- 4. Decoder (per il task di ricostruzione) ---
        # Layer per preparare lo stato nascosto iniziale del decoder da z
        self.latent_to_decoder_hidden = nn.Linear(latent_size, hidden_size * n_rnn_layer)

        # Il decoder stesso. Non è bidirezionale.
        self.decoder_rnn = nn.LSTM(
            input_size=emb_size, # L'input è solo l'embedding del token
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            dropout=dropout_rate if n_rnn_layer > 1 else 0,
        )
        self.output_to_vocab = nn.Linear(hidden_size, vocab_size)

       # --- 5. Predittore di Proprietà (per il task di regressione) ---
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_prop),  # Output delle proprietà
        )

        self._initialize_weights()

    def encode(self, x: torch.Tensor, c: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T_in = x.size()
        
        # 1. Embedding dei token
        embedded_smiles = self.embedding(x) # -> (B, T_in, embedding_dim)
        
        # 2. Condizionamento Pre-LSTM: espandi 'c' e concatena
        c_expanded = c.unsqueeze(1).repeat(1, T_in, 1) # -> (B, T_in, condition_dim)
        encoder_input = torch.cat([embedded_smiles, c_expanded], dim=2) # -> (B, T_in, embedding_dim + condition_dim)
        
        # 3. Impacchetta la sequenza e passala alla BiLSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(encoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder_rnn(packed_input)
        
        
        # 4. Concatena gli stati finali forward e backward dell'ultimo layer
        # h_n[-2] è forward, h_n[-1] è backward
        h_last_layer = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1) # -> (B, hidden_dim * 2)

        # 5. Proietta nello spazio latente
        z = self.to_latent(h_last_layer) # -> (B, latent_dim)        
        
        return z

    def decode(self, z: torch.Tensor, x_target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Ricostruisce lo SMILES a partire da z (usando teacher forcing).
        """
        # z: (B, latent_dim), x_target: (B, T_out), lengths: (B,)
        B, T_out = x_target.size()
        
        # 1. Prepara gli stati iniziali (h_0, c_0) del decoder da z
        hidden_flat = self.latent_to_decoder_hidden(z)
        h_0 = hidden_flat.view(self.decoder_rnn.num_layers, B, self.decoder_rnn.hidden_size)
        c_0 = torch.zeros_like(h_0) # Stato di cella iniziale a zero
        
        # 2. Embedding della sequenza target
        embedded_target = self.embedding(x_target)
        
        # 3. Passaggio nel decoder
        packed_target = nn.utils.rnn.pack_padded_sequence(embedded_target, lengths.cpu(), batch_first=True, enforce_sorted=False)
        dec_out_packed, _ = self.decoder_rnn(packed_target, (h_0, c_0))
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(dec_out_packed, batch_first=True, total_length=T_out)

        # 4. Proietta l'output ai logits del vocabolario
        logits = self.output_to_vocab(dec_out) # -> (B, T_out, vocab_size)
        return logits
        
    
    def _initialize_weights(self):
        """Inizializza i pesi dei layer lineari e degli embedding."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor, lengths: torch.Tensor):
        """
        Esegue il forward pass completo del modello.
        `x` è usato sia come input per l'encoder che come target per il decoder.
        """
        # 1. Ottieni la rappresentazione latente condizionata
        z = self.encode(x, c, lengths)
        
        # 2. Esegui i due task per la loss
        reconstruction_logits = self.decode(z, x, lengths)
        predicted_properties = self.property_predictor(z)
        
        return z, reconstruction_logits, predicted_properties
    
    
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

    def configure_optimizers(self, learning_rate, weight_decay=0.01, betas=(0.9, 0.999)):
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


class OLDConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_prop,
        latent_size=256,
        emb_size=128,
        hidden_size=512,
        n_rnn_layer=3,
        encoder_biLSTM=True,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.latent_size = latent_size

        # --- Embedding per token
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # --- Encoder (processa solo SMILES) ----------------------------
        self.encoder_rnn = nn.LSTM(
            input_size=emb_size, # Solo embedding
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=encoder_biLSTM,
            # dropout=0.2 if n_rnn_layer > 1 else 0,
        )

        encoder_output_size = hidden_size * (2 if encoder_biLSTM else 1)
        
        # Questo layer genera i parametri di modulazione
        # self.film_generator = nn.Linear(self.num_prop, encoder_output_size * 2)
        self.embedding_prop_output_size = encoder_output_size // 2
        # Layer per mappare le proprietà in uno spazio compatibile con l'output dell encoder
        self.embedding_prop = nn.Linear(num_prop, self.embedding_prop_output_size)
        

        # --- Latent Space Formation (fonde struttura e proprietà) ------
        # # Layer per mappare [encoder_output, proprietà] -> z
        self.to_latent = nn.Sequential(
            nn.Linear(encoder_output_size + self.embedding_prop_output_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_size)
        )
        # to_latent ora prende solo l'output modulato
        # self.to_latent = nn.Linear(encoder_output_size, latent_size)

        # --- Decoder (condizionato solo su z) ---------------------------
        # Layer per preparare lo stato nascosto iniziale del decoder da z
        self.latent_to_decoder_hidden = nn.Linear(latent_size, hidden_size * n_rnn_layer)

        self.decoder_rnn = nn.LSTM(
            input_size=emb_size, # Input: solo token precedente
            hidden_size=hidden_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=False, # Deve essere False per la generazione
            # dropout=0.2 if n_rnn_layer > 1 else 0,
        )
        self.output_linear = nn.Linear(hidden_size, vocab_size)

        # --- Property Predictor (dallo spazio latente) -------------------
        self.property_predictor = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), # <-- Aggiungi LayerNorm qui!
            nn.GELU(),
            nn.Dropout(0.25), # <-- Aggiungi Dropout qui!
            nn.Linear(hidden_size // 2, num_prop)
        )

        self._initialize_weights()

    def encode(self, x, c, lengths):
        emb = self.embedding(x)

        # 1. Encoder processa solo SMILES
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.encoder_rnn(packed)

        # 2. Estrai l'output dell'encoder
        if self.encoder_rnn.bidirectional:
            h_last_layer = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        else:
            h_last_layer = h_n[-1, :, :]
            
        # film_params = self.film_generator(c)  # (B, 2 * encoder_output_size)
        
        # # Dividi i parametri
        # gamma, beta = torch.chunk(film_params, 2, dim=-1) # Ognuno ha dim (B, encoder_output_size)

        # # Applica FiLM
        # modulated_h = gamma * h_last_layer + beta
        # # Proietta nello spazio latente
        # z = self.to_latent(modulated_h)
        
        c = self.embedding_prop(c)  # (B, embedding_prop_output_size)
        
        # 3. Fonda output encoder e proprietà per creare z
        latent_input = torch.cat([h_last_layer, c], dim=-1)
        z = self.to_latent(latent_input)
        
        return z

    def decode(self, z, x_target, lengths):
        # x_target è lo SMILES target per il teacher forcing
        B, T = x_target.size()
        emb = self.embedding(x_target)

        # 1. Usa z per inizializzare lo stato nascosto del decoder
        # Questo è un modo robusto per condizionare
        hidden_flat = self.latent_to_decoder_hidden(z)
        h_0 = hidden_flat.view(self.decoder_rnn.num_layers, B, self.decoder_rnn.hidden_size)
        c_0 = torch.zeros_like(h_0) # Inizializza a zero il cell state
        
        # 2. Il decoder riceve solo la sequenza di embedding
        # Non ha bisogno di z ad ogni step, perché è già nel suo stato iniziale
        # Nota: dobbiamo passare le lunghezze anche qui se le sequenze sono paddate
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        dec_out_packed, _ = self.decoder_rnn(packed, (h_0, c_0))
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(dec_out_packed, batch_first=True, total_length=T)

        logits = self.output_linear(dec_out)
        return logits
    
    def _initialize_weights(self):
        """Inizializza i pesi dei layer lineari e degli embedding."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, c, lengths):
        # x: input per l'encoder E target per il decoder (es. SMILES completo)
        z = self.encode(x, c, lengths)
        logits = self.decode(z, x, lengths) # Teacher forcing
        predicted_properties = self.property_predictor(z)
        return z, logits, predicted_properties
    
    
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

    def configure_optimizers(self, learning_rate, weight_decay=0.01, betas=(0.9, 0.999)):
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


