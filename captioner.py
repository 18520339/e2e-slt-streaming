import torch
import torch.nn as nn
import torch.nn.functional as F
from deformable_detr import TemporalMSDA


class DeformableLSTM(nn.Module): # A deformable version of https://arxiv.org/abs/1502.03044
    def __init__(self, config, rnn_num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.config = config
        self.n_levels = config.num_feature_levels
        self.n_heads = config.decoder_layers
        self.n_points = config.decoder_n_points
        self.deformable_attn = TemporalMSDA(config, num_heads=self.n_heads, n_points=self.n_points)

        self.attn_feat_dim = int(config.d_model / config.num_heads)
        self.attn_hidden_dim = config.d_model
        self.attn_dropout = nn.Dropout(0.5)
        self.rnn = nn.LSTM(
            input_size=config.d_model * 3,  # Input: word_embed + attn_feat
            hidden_size=config.d_model, 
            num_layers=rnn_num_layers, 
            dropout=dropout_rate, bias=False
        )
        self.ctx2attn = nn.Linear(self.attn_feat_dim, self.attn_hidden_dim)
        self.hs2attn = nn.Linear(config.d_model, self.attn_hidden_dim)
        self.alpha_net = nn.Linear(self.attn_hidden_dim, 1)
            

    def forward(
        self, token, state, query, 
        reference_points, temporal_shapes, level_start_index, 
        encoder_last_hidden_states, encoder_attention_mask
    ):
        batch_size, num_queries, n_levels, _ = reference_points.shape

        # Use (B, Q, C) for deformable attention; concat previous h with query along feature dim
        h_last = state[0][-1].view(batch_size, num_queries, -1)
        clip = self.deformable_attn(
            hidden_states=torch.cat((h_last, query), 2),  # concat features
            reference_points=reference_points, 
            temporal_shapes=temporal_shapes, 
            level_start_index=level_start_index, 
            encoder_hidden_states=encoder_last_hidden_states, 
            encoder_attention_mask=encoder_attention_mask
        )
        clip = clip.reshape(batch_size, self.n_heads, -1, num_queries, self.n_levels * self.n_points).permute(0, 3, 1, 4, 2)
        clip = clip.reshape(batch_size * num_queries, self.n_heads, self.n_levels * self.n_points, self.attn_feat_dim)
        attn_size = self.n_levels * self.n_points

        attn = self.ctx2attn(clip)                                          # (B*Q, H, A, attn_hidden_dim)
        attn = attn.view(-1, self.n_heads, attn_size, self.attn_hidden_dim) # (B*Q, H, A, attn_hidden_dim)
        attn_h = self.hs2attn(state[0][-1])                                 # (B*Q, attn_hidden_dim)
        attn_h = attn_h.unsqueeze(1).unsqueeze(1).expand_as(attn)           # (B*Q, H, A, attn_hidden_dim)
        dot = torch.tanh(attn + attn_h)                                     # (B*Q, H, A, attn_hidden_dim)
        dot = dot.view(-1, self.attn_hidden_dim)                            # (B*Q*H*A, attn_hidden_dim)
        dot = self.alpha_net(dot).view(-1, attn_size)                       # (B*Q, A)

        weight = F.softmax(dot, dim=1)
        attn_feats_ = clip.reshape(-1, attn_size, self.attn_feat_dim)       # (B*Q, A, attn_feat_dim)
        attn_res = torch.bmm(weight.unsqueeze(1), attn_feats_).squeeze(1)   # (B*Q, attn_feat_dim)
        attn_res = attn_res.reshape(batch_size * num_queries, self.n_heads, self.attn_feat_dim).flatten(1)

        # Flatten query for LSTM input
        query_flat = query.reshape(1, batch_size * num_queries, -1)         # (1, B*Q, C)
        input_feats = torch.cat((attn_res.unsqueeze(0), query_flat), 2)     # (1, B*Q, 2*C)
        output, state = self.rnn(torch.cat([token.unsqueeze(0), input_feats], 2), state)
        return output.squeeze(0) # (B*Q, d_model)


class CaptionHead(nn.Module):
    def __init__(self, config, vocab_size, rnn_num_layers, dropout_rate, max_caption_len):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.rnn_num_layers = rnn_num_layers
        self.max_caption_len = max_caption_len
        self.deformable_rnn = DeformableLSTM(config, rnn_num_layers, dropout_rate)

        self.schedule_sampling_prob = 0.25
        self.embed = nn.Embedding(self.vocab_size + 1, config.d_model)
        self.logit = nn.Linear(config.d_model, self.vocab_size + 1)
        self.dropout = nn.Dropout(dropout_rate)

        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)

        # These will be populated per call to prepare_for_captioning
        self.event_batch_idx = None
        self.event_query_idx = None


    def build_loss(self, input, target, mask): 
        # input: (num_events, max_len - 1, vocab_size + 1) or (B, Q, L, V)
        # target, mask: (num_events, max_len - 1) or (B, Q, L)
        if input.dim() == 4:  # reshape (B, Q, L, V) -> (B*Q, L, V)
            B, Q, L, V = input.shape
            input = input.view(B * Q, L, V)
            target = target.view(B * Q, L)
            mask = mask.view(B * Q, L)
            
        one_hot = F.one_hot(target, self.vocab_size + 1) # (num_events, max_len - 1, vocab_size + 1)
        max_len = input.shape[1]
        return -( # Cross entropy loss with masking
            one_hot[:, :max_len] * input * mask[:, :max_len, None]
        ).sum(2).sum(1) / (mask.sum(1) + 1e-6) # (num_events,)


    def prepare_for_captioning(self, num_queries, reference_points, valid_ratios):
        # Allocate hidden state for all sequences (B*Q) and build mapping to (B, Q)
        weight = next(self.deformable_rnn.rnn.parameters()).data
        batch_size = reference_points.shape[0]
        num_events = batch_size * num_queries
        device = reference_points.device
        state = ( # (h0, c0)
            weight.new_zeros(self.rnn_num_layers, num_events, self.config.d_model),
            weight.new_zeros(self.rnn_num_layers, num_events, self.config.d_model)
        )
        # Build mapping: for each event index e in [0, B*Q), event_batch_idx[e] = b, event_query_idx[e] = q
        b_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_queries).reshape(-1)
        q_idx = torch.arange(num_queries, device=device).unsqueeze(0).expand(batch_size, num_queries).reshape(-1)
        self.event_batch_idx = b_idx
        self.event_query_idx = q_idx

        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] * torch.stack([valid_ratios] * 2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * valid_ratios[:, None, :, None]
        return state, reference_points


    def get_log_probs(
        self, token, state, decoder_hidden_states, 
        reference_points, temporal_shapes, level_start_index, 
        encoder_last_hidden_states, encoder_attention_mask
    ):
        output = self.deformable_rnn( # batch_first=False as default in nn.LSTM
            self.embed(token), state, decoder_hidden_states, 
            reference_points, temporal_shapes, level_start_index, 
            encoder_last_hidden_states, encoder_attention_mask
        )
        output = self.logit(self.dropout(output))
        return F.log_softmax(output, dim=1) # (num_events, vocab_size + 1)

    def forward( # Teacher forcing during training
        self, seq_tokens, decoder_hidden_states, 
        reference_points, temporal_shapes, level_start_index, 
        valid_ratios, encoder_last_hidden_states, encoder_attention_mask
    ):
        batch_size, num_queries, _ = decoder_hidden_states.shape
        num_events = batch_size * num_queries
        # After this call, use self.event_batch_idx and self.event_query_idx to know which (B, Q)
        state, reference_points = self.prepare_for_captioning(num_queries, reference_points, valid_ratios)

        outputs = []
        seq_tokens = seq_tokens.long()
        if seq_tokens.dim() == 3: 
            seq_tokens = seq_tokens.view(-1, seq_tokens.size(-1))  # (B*Q, L)

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
                        tensor=torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_idx)
                    )
                token = token.detach()

            if i >= 1 and seq_tokens[:, i].data.sum() == 0: break # Break if all sequences end
            output = self.get_log_probs(
                token, state, decoder_hidden_states, 
                reference_points, temporal_shapes, level_start_index, 
                encoder_last_hidden_states, encoder_attention_mask
            )
            outputs.append(output)
        return torch.cat([output.unsqueeze(1) for output in outputs], 1) # (B*Q, max_len - 1, vocab_size + 1)
        # return out.view(batch_size, num_queries, out.size(1), out.size(2))


    @torch.no_grad()
    def sample( # Greedy or multinomial sampling during inference
        self, decoder_hidden_states, 
        reference_points, temporal_shapes, level_start_index, 
        encoder_last_hidden_states, encoder_attention_mask, 
        valid_ratios, sample_max=1, temperature=1.0
    ): 
        batch_size, num_queries, _ = decoder_hidden_states.shape
        seq_log_probs, seq_tokens = [], []
        state, reference_points = self.prepare_for_captioning(num_queries, reference_points, valid_ratios)
        num_events = batch_size * num_queries

        # Initialize with <BOS> for all events (B*Q)
        token = decoder_hidden_states.new_zeros(num_events, dtype=torch.long)
        unfinished = None

        for t in range(self.max_caption_len):
            output = self.get_log_probs(
                token, state, decoder_hidden_states, 
                reference_points, temporal_shapes, level_start_index, 
                encoder_last_hidden_states, encoder_attention_mask
            )
            if sample_max: # Greedy decoding
                step_log_probs, next_token = torch.max(output.data, 1)
                next_token = next_token.view(-1).long()
            else: # Sample from distribution
                prob_prev = torch.exp(torch.div(output.data, temperature)) # Scale output by temperature
                next_token = torch.multinomial(prob_prev, 1).view(-1).long()
                step_log_probs = output.gather(1, next_token.unsqueeze(1)).view(-1) # Gather the output at sampled positions

            unfinished = (next_token > 0) if (unfinished is None) else (unfinished & (next_token > 0)) # End token assumed to be 0
            if unfinished.sum() == 0: break # Stop when all finished
            next_token = next_token * unfinished.long() # Mask out finished sequences
            seq_log_probs.append(step_log_probs)
            seq_tokens.append(next_token)
            token = next_token # feed next token

        if len(seq_tokens) == 0: return [], []
        seq_log_probs = torch.stack(seq_log_probs, dim=1)  # (B*Q, T)
        seq_tokens = torch.stack(seq_tokens, dim=1)        # (B*Q, T)

        # Return structured (B, Q, T)
        T = seq_tokens.shape[-1]
        return seq_log_probs.view(batch_size, num_queries, T), seq_tokens.view(batch_size, num_queries, T)