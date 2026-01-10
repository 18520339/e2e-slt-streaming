import torch
import torch.nn.functional as F


def debug_variance(model, pixel_values, pixel_mask, labels=None):
    # Debug script to find where input variance is lost. Run with 2+ different inputs in a batch to check variance
    B = pixel_values.shape[0]
    print(f"{'='*60}")
    print("VARIANCE DEBUGGING - Testing if outputs differ across batch")
    
    # 1. Check input variance
    input_var = pixel_values.var(dim=0).mean()
    print(f"[1] Input pixel_values variance: {input_var.item():.6f}")
    
    # 2. Check backbone output
    backbone_out = model.transformer.backbone(pixel_values)
    if isinstance(backbone_out, tuple): backbone_out = backbone_out[0]
    backbone_out = backbone_out.permute(0, 2, 1)  # [B, d_model, T]
    backbone_var = (backbone_out[0] - backbone_out[1:]).abs().mean() if B > 1 else backbone_out.var()
    print(f"[2] Backbone output difference (B0 vs others): {backbone_var.item():.6f}")
    
    # 3. Check position embeddings
    pos_embed = model.transformer.position_embeddings(backbone_out, pixel_mask.to(torch.bool), durations=torch.sum(pixel_mask, 1))
    pos_var = (pos_embed[0] - pos_embed[1:]).abs().mean() if B > 1 else pos_embed.var()
    print(f"[3] Position embeddings difference: {pos_var.item():.6f}")
    
    # 4. Check encoder output
    with torch.no_grad():
        transformer_out = model.transformer(pixel_values, pixel_mask, labels=labels, return_dict=True)
    
    enc_hidden = transformer_out.encoder_last_hidden_state
    enc_var = (enc_hidden[0] - enc_hidden[1:]).abs().mean() if B > 1 else enc_hidden.var()
    print(f"[4] Encoder output difference: {enc_var.item():.6f}")
    
    # 5. Check decoder hidden states
    dec_hidden = transformer_out.intermediate_hidden_states[:, -1]  # Last layer
    dec_var = (dec_hidden[0] - dec_hidden[1:]).abs().mean() if B > 1 else dec_hidden.var()
    print(f"[5] Decoder hidden states difference: {dec_var.item():.6f}")
    
    # 6. Check reference points
    ref_points = transformer_out.intermediate_reference_points[:, -1]  # Last layer
    ref_var = (ref_points[0] - ref_points[1:]).abs().mean() if B > 1 else ref_points.var()
    print(f"[6] Reference points difference: {ref_var.item():.6f}")
    
    # 7. Check init reference points (before decoder)
    init_ref = transformer_out.init_reference_points
    init_var = (init_ref[0] - init_ref[1:]).abs().mean() if B > 1 else init_ref.var()
    print(f"[7] Initial reference points difference: {init_var.item():.6f}")
    
    print("\nINTERPRETATION:")
    print("- If variance drops to ~0 at step N, the bug is at/before step N")
    print("- Steps with 0 variance = batch items are identical")
    print(f"{'='*60}\n")
    
    return {
        'input': input_var.item(),
        'backbone': backbone_var.item(),
        'position': pos_var.item(),
        'encoder': enc_var.item(),
        'decoder': dec_var.item(),
        'reference_points': ref_var.item(),
        'init_reference': init_var.item(),
    }


def debug_query_variance(model, pixel_values, pixel_mask):
    # Debug variance across QUERIES (events) within each batch item
    print(f"{'='*60}")
    print("QUERY VARIANCE DEBUG - Checking if queries differ within each batch")
    
    with torch.no_grad():
        model.eval()
        out = model.transformer(pixel_values, pixel_mask, return_dict=True)
    
    # dec_hidden shape: (B, num_layers, Q, D) -> take last layer
    dec_hidden = out.intermediate_hidden_states[:, -1]  # (B, Q, D)
    B, Q, D = dec_hidden.shape
    
    # Check variance across queries within each batch
    for b in range(B):
        query_diff = (dec_hidden[b, 0:1] - dec_hidden[b, 1:]).abs().mean()
        print(f"[Batch {b}] Query variance (Q0 vs others): {query_diff.item():.6f}")
    
    # Check reference points across queries
    ref_points = out.intermediate_reference_points[:, -1]  # (B, Q, 2)
    print(f"\nReference points per query:")
    for b in range(B):
        print(f"  Batch {b}: {ref_points[b].tolist()}")
    
    # Check initial query embeddings
    query_embed = model.transformer.query_position_embeddings.weight  # (Q, 2*D)
    print(f"\nQuery embeddings variance: {query_embed.var().item():.6f}")
    query_diff = (query_embed[0:1] - query_embed[1:]).abs().mean()
    print(f"Query embeddings diff (Q0 vs others): {query_diff.item():.6f}")
    print(f"{'='*60}\n")


def debug_encoder_layers(model, pixel_values, pixel_mask):
    # Debug encoder variance step-by-step to find where variance collapses
    B = pixel_values.shape[0]
    device = pixel_values.device
    transformer = model.transformer
    config = model.config
    
    print(f"{'='*60}")
    print("ENCODER LAYER-BY-LAYER VARIANCE DEBUG")
    
    # 1. Get backbone output
    backbone_out = transformer.backbone(pixel_values)
    if isinstance(backbone_out, tuple): backbone_out = backbone_out[0]
    backbone_out = backbone_out.permute(0, 2, 1)  # [B, d_model, T]
    print(f"[1] Backbone output variance: {(backbone_out[0] - backbone_out[1:]).abs().mean().item():.6f}")
    
    # 2. Build multi-scale features like the model does
    C, T = backbone_out.shape[1], backbone_out.shape[2]
    pos_level0 = transformer.position_embeddings(backbone_out, pixel_mask, durations=torch.sum(pixel_mask, 1))
    source_level0 = transformer.input_proj[0](backbone_out)
    mask_level0 = pixel_mask.to(torch.bool)
    
    sources, masks, pos_list = [source_level0], [mask_level0], [pos_level0]
    print(f"[2] After input_proj[0] variance: {(source_level0[0] - source_level0[1:]).abs().mean().item():.6f}")
    
    for level in range(1, config.num_feature_levels):
        if level == 1:
            source = transformer.input_proj[level](backbone_out)
            base_mask = pixel_mask
        else:
            source = transformer.input_proj[level](sources[-1])
            base_mask = masks[-1]
        mask = F.interpolate(base_mask[None].float(), size=source.shape[-1:], mode='nearest').to(torch.bool)[0]
        pos_l = transformer.position_embeddings(source, mask, durations=torch.sum(mask, 1)).to(source.dtype)
        sources.append(source)
        masks.append(mask)
        pos_list.append(pos_l)
        print(f"[2] After input_proj[{level}] variance: {(source[0] - source[1:]).abs().mean().item():.6f}")
    
    # 3. Flatten and pass through encoder
    source_flatten, mask_flatten, lvl_pos_embed_flatten = [], [], []
    temporal_shapes = []
    for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, pos_list)):
        _, _, width = source.shape
        temporal_shapes.append(width)
        source_flatten.append(source.transpose(1, 2))
        mask_flatten.append(mask)
        lvl_pos_embed_flatten.append(pos_embed.transpose(1, 2) + transformer.level_embed[level].view(1, 1, -1))
    
    source_flatten = torch.cat(source_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat([temporal_shapes.new_zeros((1,)), temporal_shapes.cumsum(0)[:-1]])
    valid_ratios = torch.stack([torch.sum(m, 1).to(source_flatten.dtype) / m.shape[1] for m in masks], 1)
    
    print(f"[3] Flattened source variance: {(source_flatten[0] - source_flatten[1:]).abs().mean().item():.6f}")
    
    # 4. Call encoder with debug=True
    encoder = transformer.encoder
    reference_points = encoder.get_reference_points(temporal_shapes, valid_ratios, device=device)
    hidden_states = F.dropout(source_flatten, p=encoder.dropout, training=encoder.training)
    print(f"[4] After encoder dropout variance: {(hidden_states[0] - hidden_states[1:]).abs().mean().item():.6f}")
    
    for idx, layer in enumerate(encoder.layers):
        layer_out = layer(
            hidden_states, mask_flatten,
            position_embeddings=lvl_pos_embed_flatten,
            reference_points=reference_points,
            temporal_shapes=temporal_shapes,
            level_start_index=level_start_index,
        )
        hidden_states = layer_out[0]
        print(f"[5] After encoder layer {idx} variance: {(hidden_states[0] - hidden_states[1:]).abs().mean().item():.6f}")
    print(f"{'='*60}\n")