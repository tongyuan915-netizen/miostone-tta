import torch
import torch.nn.functional as F
from model import MIOSTONEModel


def test_time_adaptation(model: MIOSTONEModel, data_loader, steps: int = 1, lr: float = 1e-3):
    """Perform test-time adaptation on a MIOSTONE model.

    Parameters
    ----------
    model: MIOSTONEModel
        Pretrained MIOSTONE model.
    data_loader: iterable
        Dataloader yielding batches of test inputs. Labels are ignored.
    steps: int
        Number of optimization steps for each batch.
    lr: float
        Learning rate for the optimizer.

    Returns
    -------
    MIOSTONEModel
        The adapted model (modified in-place).
    """
    model.eval()

    # Collect affine parameters of all BatchNorm1d layers
    bn_params = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.train()  # enable batch statistics updating
            if m.affine:
                bn_params.append(m.weight)
                bn_params.append(m.bias)
        else:
            m.eval()

    # Freeze all parameters except BN affine parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in bn_params:
        p.requires_grad = True

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    for x, _ in data_loader:
        x = x.to(next(model.parameters()).device)
        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
            entropy.backward()
            optimizer.step()
    return model
