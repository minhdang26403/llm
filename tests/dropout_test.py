import torch

from layers.dropout import Dropout


def test_dropout():
    dropout = Dropout(p=0.5)
    x = torch.ones(1, 1000)  # 1000 ones

    # Test 1: Training Mode (Should have some zeros and be scaled to ~2.0)
    dropout.train()
    out_train = dropout(x)
    assert 0 in out_train, "Dropout failed to zero out elements"
    # The average should still be ~1.0 because of the 1/(1-p) scaling
    assert torch.isclose(out_train.mean(), torch.tensor(1.0), atol=0.1)

    # Test 2: Eval Mode (Should be identical to input)
    dropout.eval()
    out_eval = dropout(x)
    assert torch.equal(x, out_eval), "Dropout should do nothing in eval mode"
