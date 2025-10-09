import torch
import torch.nn as nn
import torch.nn.functional as F
from deformable_detr import TemporalMSDA


class DeformableLSTM(nn.Module): # A deformable version of https://arxiv.org/abs/1502.03044
    def __init__(self, config, rnn_num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.config = config
        self.n_levels = config.num_feature_levels
        self.n_heads  = config.decoder_attention_heads 
        self.n_points = config.decoder_n_points
        self.deformable_attn = TemporalMSDA(config, num_heads=self.n_heads, n_points=self.n_points, is_captioning=True)
        self.attn_feat_dim   = int(config.d_model / config.decoder_attention_heads)
        self.attn_hidden_dim = config.d_model
        self.attn_dropout    = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(
            input_size=config.d_model * 3,  # Input: word_embed + attn_feat
            hidden_size=config.d_model, num_layers=rnn_num_layers, bias=False,
            dropout=dropout_rate if rnn_num_layers > 1 else 0 # Non-zero dropout expects num_layers greater than 1
        )
        self.ctx2attn = nn.Linear(self.attn_feat_dim, self.attn_hidden_dim)
        self.hs2attn = nn.Linear(config.d_model, self.attn_hidden_dim)
        self.alpha_net = nn.Linear(self.attn_hidden_dim, 1)
            

    def forward(self, token, state, query, reference_points, transformer_outputs):
        batch_size, num_queries, n_levels, _ = reference_points.shape

        # Use the last LSTM layer hidden state; maintain both (B,Q,D) and (B*Q,D) views
        h_last = state[0][-1]                                 # (B*Q, D)
        prev_h = h_last.view(batch_size, num_queries, -1)     # (B, Q, D)
        prev_h_bq = h_last                                    # (B*Q, D)

        # Deformable attention with concatenated features along last dim
        hidden_states, attn_weights = self.deformable_attn(
            hidden_states=torch.cat((prev_h, query), 2),  # (B, Q, 2*D)
            attention_mask=transformer_outputs['mask_flatten'],
            encoder_hidden_states=transformer_outputs['encoder_last_hidden_state'], 
            reference_points=reference_points, 
            temporal_shapes=transformer_outputs['temporal_shapes'], 
            level_start_index=transformer_outputs['level_start_index'], 
        )
        hidden_states = hidden_states.reshape(batch_size, self.n_heads, -1, num_queries, self.n_levels * self.n_points).permute(0, 3, 1, 4, 2)
        hidden_states = hidden_states.reshape(batch_size * num_queries, self.n_heads, self.n_levels * self.n_points, self.attn_feat_dim)
        attn_size = self.n_levels * self.n_points

        attn = self.ctx2attn(hidden_states)                                    # (B*Q, H, A, attn_hidden_dim)
        attn = attn.view(-1, self.n_heads, attn_size, self.attn_hidden_dim)    # (B*Q, H, A, attn_hidden_dim)
        attn_h = self.hs2attn(prev_h_bq)                                       # (B*Q, attn_hidden_dim)
        attn_h = attn_h.unsqueeze(1).unsqueeze(1).expand_as(attn)              # (B*Q, H, A, attn_hidden_dim)
        dot = torch.tanh(attn + attn_h)                                        # (B*Q, H, A, attn_hidden_dim)
        dot = dot.view(-1, self.attn_hidden_dim)                               # (B*Q*H*A, attn_hidden_dim)
        dot = self.alpha_net(dot).view(-1, attn_size)                          # (B*Q*H, A)

        weight = F.softmax(dot, dim=1)
        attn_feats_ = hidden_states.reshape(-1, attn_size, self.attn_feat_dim) # (B*Q*H, A, attn_feat_dim)
        attn_res = torch.bmm(weight.unsqueeze(1), attn_feats_).squeeze(1)      # (B*Q*H, attn_feat_dim)
        attn_res = attn_res.reshape(batch_size * num_queries, self.n_heads, self.attn_feat_dim).flatten(1)  # (B*Q, D)

        # Flatten query for LSTM input
        query_flat = query.reshape(1, batch_size * num_queries, -1)                       # (1, B*Q, D)
        input_feats = torch.cat((attn_res.unsqueeze(0), query_flat), 2)                   # (1, B*Q, 2*D)
        output, state = self.rnn(torch.cat([token.unsqueeze(0), input_feats], 2), state)  # (1, B*Q, 3*D)
        return output.squeeze(0), state # (B*Q, D)


class DeformableCaptioner(nn.Module):
    def __init__(
        self, config, vocab_size, 
        bos_token_id, eos_token_id, pad_token_id,
        rnn_num_layers, dropout_rate, max_caption_len
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        self.rnn_num_layers = rnn_num_layers
        self.max_caption_len = max_caption_len
        self.deformable_rnn = DeformableLSTM(config, rnn_num_layers, dropout_rate)

        self.schedule_sampling_prob = 0.25
        self.embed = nn.Embedding(self.vocab_size, config.d_model, padding_idx=pad_token_id)
        self.logit = nn.Linear(config.d_model, self.vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)


    def prepare_for_captioning(self, num_queries, reference_points, transformer_outputs):
        # Allocate hidden state for all sequences (B*Q) and build mapping to (B, Q)
        weight = next(self.deformable_rnn.rnn.parameters()).data
        batch_size = reference_points.shape[0]
        num_events = batch_size * num_queries
        state = ( # state is a tuple of (h0, c0)
            weight.new_zeros(self.rnn_num_layers, num_events, self.config.d_model),
            weight.new_zeros(self.rnn_num_layers, num_events, self.config.d_model)
        )
        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] * torch.stack([transformer_outputs['valid_ratios']] * 2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * transformer_outputs['valid_ratios'][:, None, :, None]
        return state, reference_points


    def get_log_probs_state(self, token, state, decoder_hidden_states, reference_points, transformer_outputs):
        output, state = self.deformable_rnn( # batch_first=False as default in nn.LSTM
            self.embed(token), state, decoder_hidden_states, 
            reference_points, transformer_outputs
        )
        output = self.logit(self.dropout(output))
        return F.log_softmax(output, dim=1), state # (num_events, vocab_size + 1)


    def forward(self, seq_tokens, decoder_hidden_states, reference_points, transformer_outputs): # Teacher forcing during training
        batch_size, num_queries, _ = decoder_hidden_states.shape
        state, reference_points = self.prepare_for_captioning(num_queries, reference_points, transformer_outputs)
        num_events = batch_size * num_queries
        
        outputs, seq_tokens = [], seq_tokens.long()
        if seq_tokens.dim() == 3: 
            seq_tokens = seq_tokens.view(-1, seq_tokens.size(-1))  # (B*Q, Length)

        for i in range(seq_tokens.size(1) - 1):
            token = seq_tokens[:, i].clone() # (B*Q,)

            if self.training and i >= 1 and self.schedule_sampling_prob > 0.0:
                sample_prob = decoder_hidden_states.new_zeros(num_events).uniform_(0, 1)
                sample_mask = sample_prob < self.schedule_sampling_prob
                if sample_mask.sum() != 0:
                    sample_idx = sample_mask.nonzero(as_tuple=False).view(-1)
                    prob_prev = torch.exp(outputs[-1].data) # N x (V + 1)
                    token.index_copy_(
                        dim=0, index=sample_idx,
                        source=torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_idx)
                    )
                token = token.detach()

            if i >= 1 and ((seq_tokens[:, i] == self.pad_token_id) | (seq_tokens[:, i] == self.eos_token_id)).all(): 
                break # Break if all sequences reach <EOS> or <PAD>
            
            output, state = self.get_log_probs_state(token, state, decoder_hidden_states, reference_points, transformer_outputs)
            outputs.append(output) # (B*Q, vocab_size + 1)
        outputs = torch.cat([output.unsqueeze(1) for output in outputs], 1) # (B*Q, Length, vocab_size + 1)
        return outputs.view(batch_size, num_queries, outputs.size(1), -1) 


    @torch.no_grad() # Greedy or multinomial sampling during inference
    def sample(self, decoder_hidden_states, reference_points, transformer_outputs, sample_max=1, temperature=1.0): 
        batch_size, num_queries, _ = decoder_hidden_states.shape
        state, reference_points = self.prepare_for_captioning(num_queries, reference_points, transformer_outputs)
        num_events = batch_size * num_queries
        
        # Initialize with <BOS> for all events (B*Q)
        token = torch.full((num_events,), self.bos_token_id, dtype=torch.long, device=decoder_hidden_states.device)
        seq_log_probs = torch.full((num_events, self.max_caption_len), float('-inf'), dtype=torch.float, device=decoder_hidden_states.device)
        seq_tokens = torch.full((num_events, self.max_caption_len), self.pad_token_id, dtype=torch.long, device=decoder_hidden_states.device)
        
        seq_log_probs[:, 0] = 0.0
        seq_tokens[:, 0] = token 
        done = torch.zeros_like(token, dtype=torch.bool)  # (B*Q,)

        for t in range(1, self.max_caption_len):
            output, state = self.get_log_probs_state(token, state, decoder_hidden_states, reference_points, transformer_outputs)
            if sample_max: # Greedy decoding
                step_log_probs, next_token = torch.max(output.data, 1)
                next_token = next_token.view(-1).long()
            else: # Sample from distribution
                prob_prev = torch.exp(torch.div(output.data, temperature)) # Scale output by temperature
                next_token = torch.multinomial(prob_prev, 1).view(-1).long()
                step_log_probs = output.gather(1, next_token.unsqueeze(1)).view(-1) # Gather the output at sampled positions

            done = done | (next_token == self.eos_token_id) | (next_token == self.pad_token_id)
            if done.all(): break # Stop when all finished
            next_token = next_token * (~done).long() # Mask out finished sequences
            seq_log_probs[:, t] = step_log_probs
            seq_tokens[:, t] = next_token
            token = next_token # Feed next token

        T = seq_tokens.shape[-1] # Return structured (B, Q, T)
        return seq_log_probs.view(batch_size, num_queries, T), seq_tokens.view(batch_size, num_queries, T)