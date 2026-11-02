import torch

from quilt.model.toy import ToyBackend, ToyConfig


def logits_full(backend, tokens):
    rt = backend.load_patch(range(backend.spec.n_layers), with_embed=True, with_head=True)
    return rt.head(rt.forward("s", rt.embed(tokens, 0), 0))


def test_pipeline_split_matches_whole_model():
    b = ToyBackend(ToyConfig(hidden=32, heads=4, layers=6))
    tokens = torch.tensor([b.tokenizer.encode("patchwork")])
    whole = logits_full(b, tokens)
    cuts = [(0, 2), (2, 3), (3, 6)]
    h = None
    for i, (s, e) in enumerate(cuts):
        rt = b.load_patch(range(s, e), with_embed=(i == 0), with_head=(i == len(cuts) - 1))
        h = rt.embed(tokens, 0) if i == 0 else h
        h = rt.forward("s", h, 0)
        if i == len(cuts) - 1:
            split = rt.head(h)
    assert torch.allclose(whole, split, atol=1e-5)


def test_incremental_decode_matches_full_recompute():
    b = ToyBackend(ToyConfig(hidden=32, heads=4, layers=4))
    rt = b.load_patch(range(4), with_embed=True, with_head=True)
    t1 = torch.tensor([b.tokenizer.encode("abc")])
    rt.forward("s", rt.embed(t1, 0), 0)
    t2 = torch.tensor([b.tokenizer.encode("d")])
    inc = rt.head(rt.forward("s", rt.embed(t2, 3), 3))[:, -1]
    full = logits_full(b, torch.cat([t1, t2], 1))[:, -1]
    assert torch.allclose(inc, full, atol=1e-4)
    assert rt.session_length("s") == 4


def test_same_seed_same_weights_across_instances():
    a = ToyBackend().load_patch(range(1, 2), with_embed=False, with_head=False)
    b = ToyBackend().load_patch(range(1, 2), with_embed=False, with_head=False)
    for pa, pb in zip(a.blocks.parameters(), b.blocks.parameters()):
        assert torch.equal(pa, pb)


def test_position_mismatch_is_rejected():
    b = ToyBackend(ToyConfig(hidden=32, heads=4, layers=2))
    rt = b.load_patch(range(2), with_embed=True, with_head=True)
    t = torch.tensor([[1, 2, 3]])
    rt.forward("s", rt.embed(t, 0), 0)
    try:
        rt.forward("s", rt.embed(t, 7), 7)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a position mismatch error")
