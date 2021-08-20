import json
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/home/govind/Documents/ML/Velario Youtube/extracting_mfccs_music_genre/data.json"


def load_data(data_path):
    """Loads training dataset from json file

    Args:
        data_path (str): path to json file
    Returns:
        x (ndarray) : Inputs
        y (ndarray) : Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfccs"])
    y = np.array(data["labels"])

    return x, y


def plot_history(history):
    """Plots  accuracy/loss for training/validation set as a function of epochs

    Args:
        history (object): Training history of model
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Accuracy eval")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="lower right")
    axs[1].set_title("loss eval")


if __name__ == "__main__":
    pass
