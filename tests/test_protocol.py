import numpy as np
import torch

from quilt import protocol as P


def test_tensor_roundtrip_float_and_int():
    for x in (torch.randn(2, 3, 5), torch.arange(6, dtype=torch.int64).view(1, 6), torch.zeros(1, 1, 8, dtype=torch.float16)):
        frame = P.pack({"t": "x", "v": x})
        body = frame[4:]
        back = P.decode_tensor(P.unpack(body)["v"])
        assert back.dtype == x.dtype and torch.equal(back, x)


def test_bf16_ships_as_fp16():
    x = torch.randn(4, dtype=torch.bfloat16)
    m = P.encode_tensor(x)
    assert np.dtype(m["dtype"]) == np.float16
    assert torch.allclose(P.decode_tensor(m).float(), x.float(), atol=1e-2)


def test_addr_parse():
    a = P.Addr.parse("10.0.0.5:7700")
    assert (a.host, a.port) == ("10.0.0.5", 7700)
    assert str(a) == "10.0.0.5:7700"


def test_frame_header_length():
    frame = P.pack({"t": "ping"})
    assert int.from_bytes(frame[:4], "big") == len(frame) - 4
