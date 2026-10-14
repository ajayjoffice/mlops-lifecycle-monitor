from mlops_app.data import load_dataset, split_dataset
from mlops_app.training import evaluate_predictions


def test_public_dataset_and_reproducible_split():
    x, y = load_dataset()
    assert x.shape == (569, 30)
    assert len(y) == 569
    first = split_dataset()
    second = split_dataset()
    assert first[0].equals(second[0])
    assert first[2].equals(second[2])


def test_evaluate_predictions():
    metrics = evaluate_predictions([0, 1, 1], [0, 1, 0])
    assert metrics["accuracy"] == 2 / 3
