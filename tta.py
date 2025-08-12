import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MIOSTONEDataset, MIOSTONETree
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


def run_training_tta_pipeline(
    model_fp: str,
    model_param_fp: str,
    data_fp: str,
    meta_fp: str,
    type_fp: str,
    taxonomy_fp: str | None = None,
    output_model_fp: str | None = None,
    test_result_fp: str | None = None,
    batch_size: int = 32,
    epochs: int = 1,
    lr: float = 1e-3,
):
    """Run test-time adaptation end-to-end on disk-resident resources.

    Parameters
    ----------
    model_fp: str
        Path to the pretrained model weights.
    model_param_fp: str
        JSON file containing the model type and hyperparameters.
    data_fp: str
        Path to ``data.tsv`` holding feature counts.
    meta_fp: str
        Path to ``meta.tsv`` describing samples.
    type_fp: str
        Path to the ``type.py`` script defining the target variable.
    taxonomy_fp: str, optional
        Path to ``taxonomy.nwk``; required for ``MIOSTONEModel``.
    output_model_fp: str, optional
        Destination to save the adapted model weights.
    test_result_fp: str, optional
        File to save logits and labels from the full test set.
    batch_size: int, default=32
        Batch size for the DataLoader.
    epochs: int, default=1
        Number of adaptation steps per batch.
    lr: float, default=1e-3
        Learning rate for the optimizer.

    Returns
    -------
    MIOSTONEModel
        The adapted model.
    dict
        Dictionary containing test logits, labels and elapsed time.
    """

    if taxonomy_fp is None:
        raise ValueError("taxonomy_fp is required to instantiate MIOSTONEModel")

    # Load tree and dataset
    tree = MIOSTONETree.init_from_nwk(taxonomy_fp)
    dataset = MIOSTONEDataset.init_from_files(data_fp, meta_fp, type_fp)
    tree.prune(dataset.features)
    tree.compute_depths()
    tree.compute_indices()
    dataset.order_features_by_tree(tree)
    dataset.clr_transform()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model hyperparameters and weights
    with open(model_param_fp) as f:
        params = json.load(f)
    if params.get("Model Type") != "miostone":
        raise ValueError("Only MIOSTONEModel is supported for TTA")
    model = MIOSTONEModel(tree, dataset.num_classes, **params["Model Hparams"])
    model.load_state_dict(torch.load(model_fp))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run adaptation
    start = time.time()
    test_time_adaptation(model, loader, steps=epochs, lr=lr)
    elapsed = time.time() - start

    if output_model_fp:
        torch.save(model.state_dict(), output_model_fp)

    # Evaluate on the entire dataset
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(next(model.parameters()).device)
            logits = model(x)
            logits_list.append(logits.cpu())
            labels_list.append(y)
    test_logits = torch.cat(logits_list).tolist()
    test_labels = torch.cat(labels_list).tolist()

    result = {
        "test_logits": test_logits,
        "test_labels": test_labels,
        "time_elapsed": elapsed,
    }

    if test_result_fp:
        with open(test_result_fp, "w") as f:
            json.dump(result, f, indent=4)

    return model, result
