import torch
import torch.nn as nn
import torch.nn.functional as F
from deformable_detr import TemporalMSDA


class DeformableSoftAttention(nn.Module): # A deformable version of https://arxiv.org/abs/1502.03044
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
        clip = self.deformable_attn(
            hidden_states=torch.cat((state[0][-1].unsqueeze(0), query), 2) # Concat the first token of the last hidden state with the query, 
            reference_points=reference_points, 
            temporal_shapes=temporal_shapes, 
            level_start_index=level_start_index, 
            encoder_hidden_states=encoder_last_hidden_states, 
            encoder_attention_mask=encoder_attention_mask
        )
        clip = clip.reshape(batch_size, self.n_heads, -1, num_queries, self.n_levels * self.n_points).permute(0, 3, 1, 4, 2)
        clip = clip.reshape(batch_size * num_queries, self.n_heads, self.n_levels * self.n_points, self.attn_feat_dim)
        attn_size = self.n_levels * self.n_points

        attn = self.ctx2att(clip)                                           # (batch * attn_size) * attn_hidden_dim
        attn = attn.view(-1, self.n_heads, attn_size, self.attn_hidden_dim) # batch * attn_size * attn_hidden_dim
        attn_h = self.hs2attn(state[0][-1])                                 # batch * attn_hidden_dim
        attn_h = attn_h.unsqueeze(1).unsqueeze(1).expand_as(attn)           # batch * attn_size * attn_hidden_dim
        dot = torch.tanh(attn + attn_h)                                     # batch * attn_size * attn_hidden_dim
        dot = dot.view(-1, self.attn_hidden_dim)                            # (batch * attn_size) * attn_hidden_dim
        dot = self.alpha_net(dot).view(-1, attn_size)                       # batch * attn_size

        weight = F.softmax(dot, dim=1)
        attn_feats_ = clip.reshape(-1, attn_size, self.attn_feat_dim)       # batch * attn_size * attn_feat_dim
        attn_res = torch.bmm(weight.unsqueeze(1), attn_feats_).squeeze(1)   # batch * attn_feat_dim
        attn_res = attn_res.reshape(batch_size * num_queries, self.n_heads, self.attn_feat_dim).flatten(1)
        input_feats = torch.cat((attn_res.unsqueeze(0), query), 2)
        output, state = self.rnn(torch.cat([token.unsqueeze(0), input_feats], 2), state)
        return output.squeeze(0)


class LSTMDSACaptioner(DeformableSoftAttention):
    def __init__(self, config, vocab_size, rnn_num_layers, dropout_rate, max_caption_len):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.rnn_num_layers = rnn_num_layers
        self.max_caption_len = max_caption_len
        self.deformable_soft_attn = DeformableSoftAttention(config, rnn_num_layers, dropout_rate)

        self.schedule_sampling_prob = 0.25
        self.embed = nn.Embedding(self.vocab_size + 1, config.d_model)
        self.logit = nn.Linear(config.d_model, self.vocab_size + 1)
        self.dropout = nn.Dropout(dropout_rate)

        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)


    def build_loss(self, input, target, mask): 
        # input: (num_events, max_len - 1, vocab_size + 1)
        # target, mask: (num_events, max_len - 1)
        one_hot = F.one_hot(target, self.vocab_size + 1) # (num_events, max_len - 1, vocab_size + 1)
        max_len = input.shape[1]
        return -( # Cross entropy loss with masking
            one_hot[:, :max_len] * input * mask[:, :max_len, None]
        ).sum(2).sum(1) / (mask.sum(1) + 1e-6) # (num_events,)

    
    def prepare_for_captioning(self, num_queries, reference_points, valid_ratios):
        weight = next(self.parameters()).data
        state = ( # (h0, c0)
            weight.new(self.rnn_num_layers, num_queries, config.d_model).zero_(),
            weight.new(self.rnn_num_layers, num_queries, config.d_model).zero_()
        )
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
        output = self.deformable_soft_attn(
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
        assert batch_size == 1, 'Only support batch size 1 for decoder_hidden_states in captioning' 
        outputs, seq_tokens = [], seq_tokens.long() # (num_events, max_len)
        state, reference_points = self.prepare_for_captioning(num_queries, reference_points, valid_ratios)

        for i in range(seq_tokens.size(1) - 1):
            token = seq_tokens[:, i].clone() # We can have multiple sequences (num_events) for 1 batch item or clip
            if self.training and i >= 1 and self.schedule_sampling_prob > 0.0: # Otherwise no need to sample
                sample_prob = decoder_hidden_states.new_zeros(num_queries).uniform_(0, 1)
                sample_mask = sample_prob < self.schedule_sampling_prob
                if sample_mask.sum() != 0: # Some need to sample
                    sample_idx = sample_mask.nonzero().view(-1)
                    prob_prev = torch.exp(outputs[-1].data) # Fetch prev distribution: shape N x (M + 1)
                    token = seq_tokens[:, i].clone()
                    token.index_copy_(
                        dim=0, index=sample_idx,
                        tensor=torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_idx)
                    )
                    token = Variable(token, requires_grad=False)
            
            if i >= 1 and seq_tokens[:, i].data.sum() == 0: break # Break if all sequences end
            output = self.get_log_probs(
                token, state, decoder_hidden_states, 
                reference_points, temporal_shapes, level_start_index, 
                encoder_last_hidden_states, encoder_attention_mask
            )
            outputs.append(output)
        return torch.cat([output.unsqueeze(1) for output in outputs], 1) # (num_events, max_len - 1, vocab_size + 1)


    @torch.no_grad()
    def sample( # Greedy or multinomial sampling during inference
        self, decoder_hidden_states, 
        reference_points, temporal_shapes, level_start_index, 
        encoder_last_hidden_states, encoder_attention_mask, 
        valid_ratios, sample_max=1, temperature=1.0
    ): 
        batch_size, num_queries, _ = decoder_hidden_states.shape
        assert batch_size == 1, 'Only support batch size 1 for captioning'
        seq_log_probs, seq_tokens = [], []
        state, reference_points = self.prepare_for_captioning(num_queries, reference_points, valid_ratios)
        
        # Initialize with <BOS>
        token = decoder_hidden_states.data.new(num_queries).long().zero_()
        output = self.get_log_probs(
            token, state, decoder_hidden_states, 
            reference_points, temporal_shapes, level_start_index, 
            encoder_last_hidden_states, encoder_attention_mask
        )
        for t in range(1, self.max_caption_len):
            if sample_max: # Greedy decoding
                sample_log_probs, token = torch.max(output.data, 1)
                token = token.view(-1).long()
            else: # Sample from distribution
                prob_prev = torch.exp(torch.div(output.data, temperature)) # Scale output by temperature
                token = torch.multinomial(prob_prev, 1)
                sample_log_probs = output.gather(1, token) # Gather the output at sampled positions
                token = token.view(-1).long() # And flatten indices for downstream processing
            
            unfinished = token > 0 if t == 1 else unfinished & (token > 0) # End token assumed to be 0
            if unfinished.sum() == 0: break # Stop when all finished
            token *= unfinished.type_as(token) # Mask out finished sequences
            seq_log_probs.append(sample_log_probs.view(-1))
            seq_tokens.append(token) # seq_tokens[t] the input of t+2 time step

        if seq_tokens == [] or len(seq_tokens) == 0: return [], []
        seq_log_probs = torch.cat([token.unsqueeze(1) for token in seq_log_probs], 1)
        seq_tokens = torch.cat([token.unsqueeze(1) for token in seq_tokens], 1)
        return (
            seq_log_probs.reshape(-1, num_queries, seq_log_probs.shape[-1]),
            seq_tokens.reshape(-1, num_queries, seq_tokens.shape[-1])
        )