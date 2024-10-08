from ml_collections import ConfigDict


def get_training_args():
    """
    Returns the default training arguments

    Returns:
        ConfigDict: Training arguments
    """
    return ConfigDict(
        {
            "epochs": 50,
            "batch_size": 16,
            "device": [0],
            "learning_rate": 0.001,
            "momentum": 0.9,
            "plots": True,  # Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
            "resume": False,
        }
    )
