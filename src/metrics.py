import numpy as np
from seqeval.metrics import accuracy_score


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = accuracy_score(true_predictions, true_labels)
    return {"accuracy": results}


if __name__ == "__main__":
    """
    Test the compute_metrics function.
    """
    sample_predictions = np.array(
        [
            [[0.1, 0.7, 0.2], [0.3, 0.4, 0.3]],
            [[0.8, 0.1, 0.1], [0.7, 0.2, 0.3]],
        ]
    )
    sample_labels = np.array([[1, 0], [0, 1]])  # Dummy labels for testing

    eval_results = compute_metrics((sample_predictions, sample_labels))
    print("Evaluation Metrics:", eval_results)
