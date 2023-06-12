import matplotlib.pyplot as plt
import seaborn as sns

from metrics import jaccard_index, dice_coef

def plot_history(history, plot=["val_loss"], title="Validation Loss", ylabel="loss"):
    sns.set(style="darkgrid")
    epochs = len(history[plot[0]])
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel("# epochs")
    plt.ylabel(ylabel)
    for p in plot:
        plt.plot(range(1, epochs+1), history[p], label=p)

    plt.legend()
    plt.show()
    sns.set(style="white")


def calc_metrics(output, target):
    metrics = {
        "jaccard_index": jaccard_index(output, target),
        "dice_coef": dice_coef(output, target),
    }
    return metrics


def _test():
    download_file_from_google_drive_gdown("1GyxiMbxdRs8c4Z94980w1yHKqhMAjOT1", "dataset/test.zip")


if __name__ == "__main__":
    _test()

    