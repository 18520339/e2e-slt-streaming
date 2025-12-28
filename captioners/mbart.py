import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, DeformableDetrConfig, MBartConfig, MBartForCausalLM
from transformers.models.mbart.modeling_mbart import shift_tokens_right


class MBartDecoderCaptioner(nn.Module):
    ''' mBart Decoder-based Captioner, inspired by GFSLT-VLP: https://github.com/zhoubenjia/GFSLT-VLP
    
    This captioner uses the mBart decoder to generate captions directly from query embeddings, treating them as encoder hidden states.
    We use a simple cross-attention mechanism to attend to the decoder hidden states (queries).
    '''
    def __init__(
        self, config: DeformableDetrConfig, vocab_size: int, 
        bos_token_id: int, eos_token_id: int, pad_token_id: int,
        decoder_start_token_id: int, max_event_tokens: int, 
        dropout_rate: float, num_layers: int, # Number of mBart decoder layers
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_event_tokens = max_event_tokens
        
        # Reduce the size of MBart via vocabulary trimming using https://github.com/IamAdiSri/hf-trim       
        self.mbart_config = MBartConfig( # Create MBart configuration for decoder-only model
            vocab_size=vocab_size,
            d_model=config.d_model,
            encoder_ffn_dim=config.d_model * 4,  # Standard transformer practice
            decoder_ffn_dim=config.d_model * 4,  # Standard transformer practice
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            num_hidden_layers=num_layers,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            # max_position_embeddings=max_event_tokens + 10,  # Add some buffer
            # attention_dropout=dropout_rate,
            # activation_dropout=dropout_rate,
            activation_function='relu',
            dropout=dropout_rate,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            forced_eos_token_id=eos_token_id,
            scale_embedding=True,
        )
        self.mbart_decoder = MBartForCausalLM.from_pretrained('captioners/trimmed_mbart', config=self.mbart_config, ignore_mismatched_sizes=True)
        
        # Cross-attention projection: project concatenated visual and query features to match decoder's expected encoder hidden states
        # We concatenate visual features (from transformer_outputs) with query embeddings for richer representation
        # Input: 2*D (visual features + query embeddings), Output: D (mBart encoder hidden state dimension)
        self.visual_query_projection = nn.Linear(config.d_model * 2, config.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        
    def prepare_for_captioning(self, num_queries, reference_points, transformer_outputs):
        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] * torch.stack([transformer_outputs['valid_ratios']] * 2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * transformer_outputs['valid_ratios'][:, None, :, None]
        return reference_points


    def extract_visual_features(self, reference_points, transformer_outputs):
        ''' Extract visual features from transformer outputs using reference points.
        
        Args:
            reference_points: (B, Q, n_levels, 2) - normalized reference points
            transformer_outputs: dict containing encoder outputs
            
        Returns:
            visual_features: (B, Q, D) - aggregated visual features
        '''
        batch_size, num_queries, n_levels, _ = reference_points.shape
        
        # Get encoder hidden states - this contains the visual features from the backbone
        encoder_hidden_states = transformer_outputs['encoder_last_hidden_state']  # (B, T, D)
        temporal_shapes = transformer_outputs['temporal_shapes']  # (n_levels, 1)
        level_start_index = transformer_outputs['level_start_index']  # (n_levels,)
        
        # Extract features at each level based on reference points
        visual_features_per_level = []
        for level in range(n_levels):
            # Get the temporal length for this level
            temporal_len = temporal_shapes[level].item()
            start_idx = level_start_index[level].item()
            
            # Get encoder features for this level
            level_features = encoder_hidden_states[:, start_idx:start_idx + temporal_len, :]  # (B, T_level, D)
            
            # Get reference points for this level (normalized coordinates)
            ref_points = reference_points[:, :, level, 0]  # (B, Q) - temporal coordinate
            
            # Convert normalized coordinates to indices (0 to temporal_len-1)
            indices = (ref_points * (temporal_len - 1)).long().clamp(0, temporal_len - 1)  # (B, Q)
            
            # Gather features at reference points and expand indices to (B, Q, D) for gathering
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.config.d_model)  # (B, Q, D)
            gathered_features = torch.gather(level_features, 1, indices_expanded)  # (B, Q, D)
            visual_features_per_level.append(gathered_features)
        
        # Concatenate features from all levels and average
        visual_features = torch.stack(visual_features_per_level, dim=2)  # (B, Q, n_levels, D)
        return visual_features.mean(dim=2)  # (B, Q, D) - average across levels
    

    def forward(self, seq_tokens, decoder_hidden_states, reference_points, transformer_outputs):
        ''' Forward pass with teacher forcing during training.
        
        Args:
            seq_tokens: (B, Q, L) or (B*Q, L) - ground truth token sequences without BOS
            decoder_hidden_states: (B, Q, D) - query embeddings from DETR decoder
            reference_points: (B, Q, 2) - reference points for extracting visual features
            transformer_outputs: dict - outputs from transformer containing encoder hidden states
            
        Returns:
            outputs: (B, Q, L-1, vocab_size) - predicted logits for next tokens
        '''
        batch_size, num_queries, _ = decoder_hidden_states.shape
        if seq_tokens.dim() == 3: seq_tokens = seq_tokens.view(-1, seq_tokens.size(-1))  # (B*Q, L)
        num_events = batch_size * num_queries
        
        # Prepare reference points if needed (normalize by valid ratios)
        reference_points = self.prepare_for_captioning(num_queries, reference_points, transformer_outputs)
        
        # Extract visual features using reference points
        visual_features = self.extract_visual_features(reference_points, transformer_outputs)  # (B, Q, D)
        
        # Concatenate visual features with query embeddings for richer representation
        combined_features = torch.cat([visual_features, decoder_hidden_states], dim=-1)  # (B, Q, 2*D)
        
        # Project concatenated features to encoder hidden state dimension
        encoder_hidden_states = self.visual_query_projection(combined_features)  # (B, Q, D)
        encoder_hidden_states = encoder_hidden_states.view(num_events, 1, -1)  # (B*Q, 1, D)
        encoder_attention_mask = torch.ones(num_events, 1, device=decoder_hidden_states.device, dtype=torch.long)  # (B*Q, 1)
        
        # shift_tokens_right shifts: [token1, token2, ..., EOS] -> [decoder_start, token1, token2, ...]
        input_ids = shift_tokens_right(seq_tokens, self.pad_token_id) # Input tokens: all tokens except the last one (for teacher forcing)
        attention_mask = (input_ids != self.pad_token_id).long() # Create attention mask for input tokens (1 for real tokens, 0 for padding)

        # Forward through MBart decoder
        outputs = self.mbart_decoder(
            input_ids=input_ids, # (B*Q, L)
            attention_mask=attention_mask, # (B*Q, L)
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, # Attend to all queries
            return_dict=True,
        )
        seq_log_probs = F.log_softmax(outputs.logits, dim=-1) # (B*Q, L, vocab_size)
        return seq_log_probs.view(batch_size, num_queries, seq_log_probs.size(1), -1) # (B, Q, L, vocab_size)


    @torch.no_grad()
    def sample(
        self, decoder_hidden_states, reference_points, transformer_outputs,
        sample_max: bool = True, temperature: float = 1.0, num_beams: int = 1, 
        top_k: Optional[int] = None, top_p: Optional[float] = None,
    ):
        ''' Generate captions using HuggingFace's generate method.
        
        Args:
            decoder_hidden_states: (B, Q, D) - query embeddings from DETR decoder
            reference_points: (B, Q, n_levels, 2) - reference points for extracting visual features
            transformer_outputs: dict - transformer outputs containing encoder hidden states
            sample_max: if 1, use greedy decoding; if 0, use sampling
            temperature: sampling temperature
            num_beams: number of beams for beam search
            top_k: top-k sampling parameter
            top_p: nucleus sampling parameter
            
        Returns:
            seq_log_probs: (B, Q, L) - log probabilities of generated sequences
            seq_tokens: (B, Q, L) - generated token sequences
        '''
        batch_size, num_queries, _ = decoder_hidden_states.shape
        num_events = batch_size * num_queries
        
        # Prepare reference points if needed (normalize by valid ratios)
        reference_points = self.prepare_for_captioning(num_queries, reference_points, transformer_outputs)
        
        # Extract visual features using reference points
        visual_features = self.extract_visual_features(reference_points, transformer_outputs)  # (B, Q, D)
        
        # Concatenate visual features with query embeddings
        combined_features = torch.cat([visual_features, decoder_hidden_states], dim=-1)  # (B, Q, 2*D)
        
        # Project concatenated features to encoder hidden state dimension
        encoder_hidden_states = self.visual_query_projection(combined_features)  # (B, Q, D)
        encoder_hidden_states = encoder_hidden_states.view(num_events, 1, -1)  # (B*Q, 1, D)
        encoder_attention_mask = torch.ones(num_events, 1, device=decoder_hidden_states.device, dtype=torch.long)
        
        # Generate using HuggingFace's generate method
        generation_outputs = self.mbart_decoder.generate(
            inputs_embeds=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            max_new_tokens=self.max_event_tokens,
            do_sample=not sample_max,
            temperature=temperature if not sample_max else 1.0,
            num_beams=num_beams if not sample_max else 1, 
            top_k=top_k if not sample_max else None, 
            top_p=top_p if not sample_max else None,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            return_dict_in_generate=True, output_scores=True,
        )

        # Get the raw generated sequences and scores
        raw_sequences = generation_outputs.sequences  # (B*Q, generated_len) includes decoder_start_token
        raw_scores = generation_outputs.scores        # tuple of (B*Q, vocab_size) for each generated token
        num_generated = len(raw_scores)               # Number of tokens actually generated (excluding start token)
        
        # Compute log probs only for the tokens that were actually generated
        if num_generated > 0:
            scores_tensor = torch.stack(raw_scores, dim=1)  # (B*Q, num_generated, vocab_size)
            scores_log_probs = F.log_softmax(scores_tensor, dim=-1)
            
            # Get the actual generated tokens (excluding decoder_start_token) and gather log probs for them
            generated_tokens = raw_sequences[:, 1:1+num_generated]  # (B*Q, num_generated)
            seq_log_probs = scores_log_probs.gather(2, generated_tokens.unsqueeze(-1)).squeeze(-1)  # (B*Q, num_generated)
        else:
            seq_log_probs = torch.zeros(num_events, 0, device=decoder_hidden_states.device)
        
        # Pad or truncate seq_tokens to max_event_tokens for consistency
        if raw_sequences.size(1) < self.max_event_tokens:
            padding = torch.full(
                (num_events, self.max_event_tokens - raw_sequences.size(1)), self.pad_token_id,
                dtype=torch.long, device=decoder_hidden_states.device
            )
            seq_tokens = torch.cat([raw_sequences, padding], dim=1)
        else:
            seq_tokens = raw_sequences[:, :self.max_event_tokens]
        
        # Build full log probs: [0 for start token] + [generated log probs] + [-inf for padding]
        bos_log_probs = torch.zeros(num_events, 1, device=decoder_hidden_states.device)
        seq_log_probs = torch.cat([bos_log_probs, seq_log_probs], dim=1)  # (B*Q, 1 + num_generated)
        
        # Pad or truncate seq_log_probs to max_event_tokens for consistency
        if seq_log_probs.size(1) < self.max_event_tokens:
            padding = torch.full(
                (num_events, self.max_event_tokens - seq_log_probs.size(1)), float('-inf'), 
                dtype=seq_log_probs.dtype, device=decoder_hidden_states.device
            )
            seq_log_probs = torch.cat([seq_log_probs, padding], dim=1)
        else:
            seq_log_probs = seq_log_probs[:, :self.max_event_tokens]
        
        # Return structured (B, Q, L)
        return seq_log_probs.view(batch_size, num_queries, self.max_event_tokens), seq_tokens.view(batch_size, num_queries, self.max_event_tokens)