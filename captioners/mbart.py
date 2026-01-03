import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import AutoTokenizer, DeformableDetrConfig, MBartConfig, MBartForCausalLM
from transformers.models.mbart.modeling_mbart import shift_tokens_right


class MBartDecoderCaptioner(nn.Module):
    ''' mBart Decoder-based Captioner, inspired by GFSLT-VLP: https://github.com/zhoubenjia/GFSLT-VLP
    
    This captioner uses the mBart decoder to generate captions directly from query embeddings.
    We extract a local window of visual features around each query's reference point, preserving
    temporal context while keeping it focused. A learnable query token is prepended to guide attention.
    
    encoder_hidden_states shape: (B*Q, 1 + window_size, D)
    - Position 0: Query context token (learnable projection of query embedding)
    - Positions 1 to window_size: Local visual features around the reference point
    '''
    def __init__(
        self, config: DeformableDetrConfig, vocab_size: int, 
        bos_token_id: int, eos_token_id: int, pad_token_id: int,
        decoder_start_token_id: int, max_event_tokens: int, 
        dropout_rate: float, num_layers: int, # Number of mBart decoder layers
        window_size: int = 8,  # Number of visual tokens to sample around reference point
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_event_tokens = max_event_tokens
        self.window_size = window_size
        
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
            activation_function='relu',
            dropout=dropout_rate,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            forced_eos_token_id=eos_token_id,
            scale_embedding=True,
            add_cross_attention=True, # Enable cross-attention to attend to visual features
        )
        self.mbart_decoder = MBartForCausalLM.from_pretrained('captioners/trimmed_mbart', config=self.mbart_config, ignore_mismatched_sizes=True)
        
        # Query token is prepended to the visual window to guide MBart's cross-attention
        self.query_visual_fusion = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.window_pos_embed = nn.Parameter(torch.zeros(1, window_size, config.d_model)) # Positional encoding for the window tokens (learnable)
        
        
    def prepare_for_captioning(self, num_queries, reference_points, transformer_outputs): # Prepare reference points by scaling with valid ratios
        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] * torch.stack([transformer_outputs['valid_ratios']] * 2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * transformer_outputs['valid_ratios'][:, None, :, None]
        return reference_points


    def extract_windowed_features(self, decoder_hidden_states, reference_points, transformer_outputs):
        ''' Extract a local window of visual features around each query's reference point.
        
        Args:
            decoder_hidden_states: (B, Q, D) - query embeddings from DETR decoder
            reference_points: (B, Q, n_levels, 2) - normalized reference points (center, width)
            transformer_outputs: dict containing encoder outputs
            
        Returns:
            encoder_hidden_states: (B*Q, 1 + window_size, D) - query token + windowed visual features
            encoder_attention_mask: (B*Q, 1 + window_size) - attention mask
        '''
        batch_size, num_queries, D = decoder_hidden_states.shape
        num_events = batch_size * num_queries
        
        # Get encoder hidden states (full temporal sequence)
        encoder_memory = transformer_outputs['encoder_last_hidden_state']  # (B, T, D)
        T = encoder_memory.shape[1]
        
        # Use the first level's reference point center for window extraction
        # reference_points: (B, Q, n_levels, 2) where last dim is (center, width)
        centers = reference_points[:, :, 0, 0]  # (B, Q) - use first level's center
        
        # Create sampling offsets: linearly spaced points within the window
        # Window spans from center - half_width to center + half_width
        half_width = 0.5 / self.config.num_feature_levels  # Adaptive based on feature levels
        offsets = torch.linspace(-half_width, half_width, self.window_size, device=encoder_memory.device)  # (window_size,)
        
        # Compute sample positions for each query
        # centers: (B, Q), offsets: (window_size,) -> sample_positions: (B, Q, window_size)
        sample_positions = centers.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)  # (B, Q, window_size)
        sample_positions = sample_positions.clamp(0.0, 1.0)  # Clamp to valid range
        
        # Convert normalized positions to indices
        sample_indices = (sample_positions * (T - 1)).long() # (B, Q, window_size)
        sample_indices = sample_indices.clamp(0, T - 1)
        
        # Gather features at sample positions
        # encoder_memory: (B, T, D), sample_indices: (B, Q, window_size)
        sample_indices_flat = sample_indices.view(batch_size, -1)                                 # (B, Q * window_size)
        sample_indices_expanded = sample_indices_flat.unsqueeze(-1).expand(-1, -1, D)             # (B, Q * window_size, D)
        gathered_features = torch.gather(encoder_memory, 1, sample_indices_expanded)              # (B, Q * window_size, D)
        windowed_features = gathered_features.view(batch_size, num_queries, self.window_size, D)  # (B, Q, window_size, D)
        windowed_features = windowed_features + self.window_pos_embed                             # (B, Q, window_size, D)
        
        # Combine query context token with windowed features and apply fusion
        query_context = decoder_hidden_states.unsqueeze(2)                                        # (B, Q, 1, D)
        encoder_hidden_states = torch.cat([query_context, windowed_features], dim=2)              # (B, Q, 1 + window_size, D)
        encoder_hidden_states = self.query_visual_fusion(encoder_hidden_states)                   # (B, Q, 1 + window_size, D)
        encoder_hidden_states = encoder_hidden_states.view(num_events, 1 + self.window_size, D)   # (B*Q, 1 + window_size, D)
        
        # Create attention mask (all ones since all positions are valid)
        encoder_attention_mask = torch.ones(num_events, 1 + self.window_size, device=encoder_memory.device, dtype=torch.long)
        return encoder_hidden_states, encoder_attention_mask
    

    def forward(self, seq_tokens, decoder_hidden_states, reference_points, transformer_outputs):
        ''' Forward pass with teacher forcing during training.
        
        Args:
            seq_tokens: (B, Q, L) or (B*Q, L) - ground truth token sequences without BOS
            decoder_hidden_states: (B, Q, D) - query embeddings from DETR decoder
            reference_points: (B, Q, 2) - reference points for extracting visual features
            transformer_outputs: dict - outputs from transformer containing encoder hidden states
            
        Returns:
            outputs: (B, Q, L, vocab_size) - predicted logits for next tokens
        '''
        batch_size, num_queries, _ = decoder_hidden_states.shape
        if seq_tokens.dim() == 3: seq_tokens = seq_tokens.view(-1, seq_tokens.size(-1))  # (B*Q, L)
        
        # Prepare reference points (normalize by valid ratios)
        reference_points = self.prepare_for_captioning(num_queries, reference_points, transformer_outputs)
        
        # Extract windowed visual features with query context
        encoder_hidden_states, encoder_attention_mask = self.extract_windowed_features(
            decoder_hidden_states, reference_points, transformer_outputs
        ) # (B*Q, 1 + window_size, D), (B*Q, 1 + window_size)
        
        # shift_tokens_right shifts: [token1, token2, ..., EOS] -> [decoder_start, token1, token2, ...]
        input_ids = shift_tokens_right(seq_tokens, self.pad_token_id) # Input tokens: all tokens except the last one (for teacher forcing)
        attention_mask = (input_ids != self.pad_token_id).long() # Create attention mask for input tokens (1 for real tokens, 0 for padding)

        # Forward through MBart decoder
        outputs = self.mbart_decoder(
            input_ids=input_ids,                                                      # (B*Q, L)
            attention_mask=attention_mask,                                            # (B*Q, L)
            encoder_hidden_states=encoder_hidden_states,                              # (B*Q, 1 + window_size, D)
            encoder_attention_mask=encoder_attention_mask,                            # (B*Q, 1 + window_size)
            return_dict=True,
        )
        seq_log_probs = F.log_softmax(outputs.logits, dim=-1)                         # (B*Q, L, vocab_size)
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
            reference_points: (B, Q, 2) or (B, Q, n_levels, 2) - reference points
            transformer_outputs: dict - transformer outputs containing encoder hidden states
            sample_max: if True, use greedy decoding; if False, use sampling
            temperature: sampling temperature
            num_beams: number of beams for beam search
            top_k: top-k sampling parameter
            top_p: nucleus sampling parameter
            
        Returns:
            seq_log_probs: (B, Q, L) - log probabilities of generated sequences
            seq_tokens: (B, Q, L) - generated token sequences
        '''
        batch_size, num_queries, D = decoder_hidden_states.shape
        num_events = batch_size * num_queries
        
        # Prepare reference points (normalize by valid ratios)
        reference_points = self.prepare_for_captioning(num_queries, reference_points, transformer_outputs)
        
        # Extract windowed visual features with query context
        encoder_hidden_states, encoder_attention_mask = self.extract_windowed_features(
            decoder_hidden_states, reference_points, transformer_outputs
        ) # (B*Q, 1 + window_size, D), (B*Q, 1 + window_size)
        
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