import torch
from torch import nn

from make_it_flash.hybrid_cache import FlashNextDynamicCache
from make_it_flash.flash_next import (
    CompactNGramEmbedding,
    CompactPLELayer,
    GatedResidualMixer,
    GatedSharedExpert,
    MultiTokenPredictionHead,
    QSAIndexer,
    QSAQwen3MoeAttentionAdapter,
)


def test_flash_next_cache_keeps_indexer_state_aligned_through_beams_and_reset():
    cache = FlashNextDynamicCache(["indexed_attention"])
    indexer_keys = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    position_ids = torch.arange(3).expand(2, -1)
    key_values = torch.arange(24, dtype=torch.float32).view(2, 1, 3, 4)
    cache.update_indexer(indexer_keys, position_ids, 0)
    cache.update(key_values, key_values, 0)

    beam_indices = torch.tensor([1, 0, 1])
    cache.reorder_cache(beam_indices)
    layer = cache.layers[0]
    torch.testing.assert_close(layer.indexer_keys, indexer_keys.index_select(0, beam_indices))
    torch.testing.assert_close(layer.indexer_position_ids, position_ids.index_select(0, beam_indices))
    assert cache.get_seq_length() == 3

    cache.crop(2)
    assert cache.get_seq_length() == 2
    assert layer.indexer_keys.shape[1] == 2
    cache.reset()
    assert cache.get_seq_length() == 0
    assert layer.indexer_keys is None and layer.indexer_position_ids is None


def test_qsa_indexer_selects_visible_micro_blocks_and_keeps_tail():
    torch.manual_seed(47)
    indexer = QSAIndexer(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        token_budget=4,
        compress_ratio=2,
        rotary_dim=2,
        rope_theta=100.0,
    )
    hidden = torch.randn(1, 6, 8)
    positions = torch.arange(6).unsqueeze(0)
    causal = torch.tril(torch.ones(1, 1, 6, 6, dtype=torch.bool))

    selected = indexer(hidden, positions, causal)

    assert selected.shape == (1, 1, 6, 6)
    assert selected[0, 0, 4].sum().item() == 5  # two full blocks plus the incomplete tail token
    assert selected[0, 0, 5].sum().item() == 4  # budget selects two complete blocks
    assert not selected[0, 0, 5, 6:].any()
    assert torch.equal(selected & ~causal, torch.zeros_like(selected))


def test_qsa_selection_auxiliary_loss_trains_the_discrete_indexer():
    torch.manual_seed(51)
    indexer = QSAIndexer(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        token_budget=4,
        compress_ratio=2,
        rotary_dim=2,
        rope_theta=100.0,
    )
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    positions = torch.arange(6).unsqueeze(0)
    causal = torch.tril(torch.ones(1, 1, 6, 6, dtype=torch.bool))
    teacher = torch.softmax(torch.randn(1, 2, 6, 6).masked_fill(~causal, -1e4), dim=-1)

    loss = indexer.selection_loss(hidden, positions, causal, teacher)
    loss.backward()

    assert torch.isfinite(loss)
    assert indexer.index_qk_proj.weight.grad is not None
    assert torch.isfinite(indexer.index_qk_proj.weight.grad).all()
    assert indexer.index_qk_proj.weight.grad.abs().sum() > 0


def test_qsa_attention_adapter_runs_sparse_prefill_and_backpropagates():
    torch.manual_seed(53)
    attention = QSAQwen3MoeAttentionAdapter(
        hidden_size=8,
        num_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rotary_dim=4,
        rope_theta=100.0,
        index_n_heads=2,
        index_head_dim=4,
        token_budget=4,
        compress_ratio=2,
    )
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    positions = torch.arange(6).unsqueeze(0)
    causal = torch.tril(torch.ones(1, 1, 6, 6, dtype=torch.bool))

    output, weights = attention(hidden, attention_mask=causal, position_ids=positions, output_attentions=True)
    output.square().mean().backward()

    assert output.shape == hidden.shape
    assert weights.shape == (1, 4, 6, 6)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    # Discrete top-k selection has no direct gradient; train its indexer with selection_loss.
    assert attention.indexer.index_qk_proj.weight.grad is None
    assert attention.q_proj.weight.grad is not None
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_gated_residual_mixer_outputs_stream_gates_and_injects():
    torch.manual_seed(31)
    mixer = GatedResidualMixer(hidden_size=8, stream_count=4, low_rank=6)
    stream_state = torch.randn(2, 3, 32)

    mixed, original, injection_weights = mixer(stream_state)
    branch_output = torch.randn(2, 3, 8)
    injected = mixer.inject(original, branch_output, injection_weights)
    expected = original + (branch_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)

    assert mixed.shape == (2, 3, 8)
    assert original is stream_state
    assert injection_weights.shape == (2, 3, 4)
    assert torch.all(injection_weights > 0)
    assert torch.all(injection_weights < 2)
    torch.testing.assert_close(injected, expected)


def test_gated_residual_final_mixer_collapses_streams():
    mixer = GatedResidualMixer(hidden_size=8, stream_count=4, low_rank=6, use_combine=False)
    collapsed = mixer(torch.randn(1, 5, 32))
    assert isinstance(collapsed, torch.Tensor)
    assert collapsed.shape == (1, 5, 8)


def test_compact_ngram_hashes_are_deterministic_and_reset_at_eos():
    ngram = CompactNGramEmbedding(
        vocab_size=101,
        embedding_dim=32,
        ngram_size=3,
        heads_per_ngram=2,
        bucket_count=31,
        eos_token_id=2,
        layer_index=1,
        padding_multiple=16,
    )
    tokens = torch.tensor([[7, 8, 2, 9]])
    full_hash = ngram.hash_ids(tokens)
    short_hash = ngram.hash_ids(torch.tensor([[2, 9]]))

    assert full_hash.shape == (1, 4, 4)
    assert ngram(tokens).shape == (1, 4, 32)
    torch.testing.assert_close(full_hash[:, 3], short_hash[:, 1])
    assert ngram.update_context(tokens).tolist() == [[2, 9]]
    empty_ids = torch.empty((2, 0), dtype=torch.long)
    assert ngram.hash_ids(empty_ids).shape == (2, 0, 4)
    assert ngram(empty_ids).shape == (2, 0, 32)


def test_compact_ple_is_neutral_at_init_and_trainable():
    torch.manual_seed(41)
    ple = CompactPLELayer(
        hidden_size=8,
        stream_count=4,
        ple_embed_dim=16,
        vocab_size=101,
        eos_token_id=2,
        heads_per_ngram=2,
        bucket_count=31,
        conv_kernel_size=2,
    )
    states = torch.randn(2, 4, 32, requires_grad=True)
    tokens = torch.randint(0, 101, (2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    initial = ple(states, tokens, mask)
    assert initial.shape == states.shape
    torch.testing.assert_close(initial, torch.zeros_like(initial))

    with torch.no_grad():
        ple.value_proj.weight.fill_(0.01)
    output = ple(states, tokens, mask)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert ple.value_proj.weight.grad is not None


def test_shared_expert_adapter_is_neutral_and_additive():
    torch.manual_seed(43)
    shared = GatedSharedExpert(hidden_size=8, intermediate_size=4)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    routed = torch.randn(2, 3, 8)

    initial = shared(hidden)
    torch.testing.assert_close(initial, torch.zeros_like(initial))
    with torch.no_grad():
        shared.down_proj.weight.normal_(mean=0.0, std=0.01)
    result = shared.add_to(routed, hidden)
    torch.testing.assert_close(result - routed, shared(hidden))
    result.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_mtp_head_uses_tied_embeddings_and_future_targets():
    torch.manual_seed(37)
    embedding = nn.Embedding(17, 12)
    head = MultiTokenPredictionHead(hidden_size=12, vocab_size=17, horizon=2)
    hidden_states = torch.randn(2, 5, 12, requires_grad=True)
    input_ids = torch.randint(0, 17, (2, 5))

    logits = head(hidden_states, embedding.weight)
    loss = head.loss(logits, input_ids)
    loss.backward()

    assert logits.shape == (2, 2, 5, 17)
    assert torch.isfinite(loss)
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert embedding.weight.grad is not None and torch.isfinite(embedding.weight.grad).all()
    assert head.projections[0].weight.grad is not None
