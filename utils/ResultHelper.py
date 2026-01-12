from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score, specificity_score
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os


def confirm_path(path, name, extension):
    if extension[0] == '.':
        extension = extension[1:]
    if path[-1] == '/':
        fixed_path = f"{path}{name}.{extension}"
    else:
        fixed_path = f"{path}/{name}.{extension}"
    return fixed_path

class ResultHelper:
    def __init__(self, name="model", folder_path=""):
        self.name = name
        self.folder_path = folder_path
        self.metrics = ["Acc", "Rec", "Prec", "F1", "BAC", "G-mean", "Spec"]
        self.scores = {"Acc": [],
                       "Rec": [],
                       "Prec": [],
                       "F1": [],
                       "BAC": [],
                       "G-mean": [],
                       "Spec": []}

    def append_all_scores(self, y_test, y_pred):
        self.scores[self.metrics[0]].append(accuracy_score(y_test, y_pred))
        self.scores[self.metrics[1]].append(recall_score(y_test, y_pred))
        self.scores[self.metrics[2]].append(precision_score(y_test, y_pred))
        self.scores[self.metrics[3]].append(f1_score(y_test, y_pred))
        self.scores[self.metrics[4]].append(balanced_accuracy_score(y_test, y_pred))
        self.scores[self.metrics[5]].append(geometric_mean_score(y_test, y_pred))
        self.scores[self.metrics[6]].append(specificity_score(y_test, y_pred))

    def print_all_scores(self):
        for metric in self.metrics:
            print(f"{metric} = {self.scores[metric]}")

    def print_mean_scores(self):
        for metric in self.metrics:
            print(f"{metric} = {round(np.mean(self.scores[metric]), 3)}")

    def save_scores(self):
        path = f"{self.folder_path}/{self.name}.csv"
        path = os.path.normpath(path)
        file = open(path, "w")
        for metric in self.metrics:
            file.write(f"{metric}")
            for score in self.scores[metric]:
                file.write(f",{score}")
            file.write("\n")
        file.close()

    def plot_radar_chart_for_ml_models(self):
        means = []
        for metric in self.metrics:
            means.append(np.mean(self.scores[metric]))

        N = len(self.scores)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], self.metrics)

        ax.set_rlabel_position(0)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                   color="grey", size=7)
        plt.ylim(0,1)

        means += means[:1]
        ax.plot(angles, means, linewidth=1, linestyle='solid')

        # plt.title(f"{self.name} Performance Chart", size=20, color='black', y=1.06)
        # path = f"./results/radar_{self.name}.png"
        path = f"{self.folder_path}/radar_{self.name}.png"
        path = os.path.normpath(path)
        plt.savefig(path, dpi=200)
        plt.show()

    @staticmethod
    def plot_radar_combined(models: dict, folder_path):
        metrics = ["G-mean", "Acc", "Rec", "Prec", "F1", "BAC", "Spec"]

        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], metrics)

        ax.set_rlabel_position(0)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                   ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                   color="grey", size=7)
        plt.ylim(0,1)

        for name, scores in models.items():
            values = []
            for metric in metrics:
                values.append(np.mean(scores[metric]))
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=name)

        # plt.title("Combined Performance Chart", size=20, color='black', y=1.06)
        path = f"{folder_path}/radar_combined.png"
        path = os.path.normpath(path)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.show()

    def plot_data_stream(self, data_stream, name="data_stream", save_path=None, start=0, end=-1):
        if end == -1:
            end = len(data_stream)
        x_values = range(start, end)
        data_to_plot = data_stream
        try:
            data_to_plot = data_stream[start:end]
        except Exception:
            print("COULDNT CROP PLOT")

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, data_to_plot, marker='None', color='r')
        plt.xlabel('Numer bloku danych')
        plt.ylabel('Stopień niezbalansowania')
        plt.grid(True)

        if save_path is not None:
            path = confirm_path(save_path, name, 'png')
            path = os.path.normpath(path)
            plt.savefig(path, dpi=200)

        plt.show()

    def append_batch_scores(self, batch_scores):
        for metric, score in batch_scores.items():
            if metric in self.scores:
                self.scores[metric].append(score)

    def plot_metrics_per_batch(self, batch_metrics):
        for metric, scores in batch_metrics.items():
            plt.figure(figsize=(8, 6))
            plt.plot(scores, label=metric)
            plt.title(f'{metric} per Batch')
            plt.ylim(0, 1)
            plt.xlabel("Batch")
            plt.ylabel(metric)
            plt.legend()
            plt.show()


def plot_class_distribution(y, folder_path):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    # Calculate percentages
    percentages = {label: (count / total) * 100 for label, count in zip(unique, counts)}
    class_names = ["Zdrowe" if label == 0 else "Chore" for label in percentages.keys()]

    plt.figure(figsize=(6, 4))
    plt.bar(class_names, percentages.values(),
            color=['skyblue', 'salmon'], edgecolor='black', alpha=0.8)
    plt.ylim(0, 100)
    plt.ylabel("Procent (%)")

    # Add percentage + count above bars
    for i, (percentage, count) in enumerate(zip(percentages.values(), counts)):
        plt.text(
            i,
            percentage + 1,
            f"{percentage:.2f}%\n(n={count})",
            ha='center',
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(f"{folder_path}/class_distribution.png")
    plt.show()

def print_mean_results(path):
    file = open(path, 'r')
    data = file.read()
    rows = data.strip().split("\n")

    metrics = ['Acc', 'Rec', 'Prec', 'Spec', 'F1', 'BAC', 'G-mean']
    results = {metric: [] for metric in metrics}

    for row in rows:
        columns = row.split(",")

        metric_name = columns[0]
        values = list(map(float, columns[1:]))

        if metric_name in results:
            results[metric_name].extend(values)

    for metric, values in results.items():
        mean_value = np.mean(values)
        print(f"Mean {metric}: {mean_value:.3f}")

    file.close()


def get_values(path):
    file = open(path, 'r')
    data = file.read()
    rows = data.strip().split("\n")

    results = {}
    for row in rows:
        columns = row.split(",")

        metric_name = columns[0]
        values = list(map(float, columns[1:]))
        results[metric_name] = values

    file.close()
    return results
