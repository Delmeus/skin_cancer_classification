# from statistical_Test import perform_test
# from utils.ResultHelper import get_values
# from utils.ResultHelper import ResultHelper
# from tabulate import tabulate
# from numpy import mean
#
# dataset = "cnn"
# rh = ResultHelper()
# metrics = ['Acc', 'Rec', 'Prec', 'Spec', 'F1', 'BAC', 'G-mean']
# knn = get_values(f"../results/e1/{dataset}/kNN_{dataset}.csv")
# nb = get_values(f"../results/e1/{dataset}/NB_{dataset}.csv")
# svm = get_values(f"../results/e1/{dataset}/SVC_{dataset}.csv")
# rl = get_values(f"../results/e1/{dataset}/LR_{dataset}.csv")
#
# models = {
#     "kNN": knn,
#     "NB": nb,
#     "RL": rl,
#     "SVC": svm,
# }
#
# rh.plot_radar_combined(models, "./")
#
#
#
# def gather_and_print_metrics(metrics):
#     """
#     Collects metrics from all models, computes mean values,
#     and prints a formatted table of results.
#     """
#     # Prepare the table header
#     header = ["Metric", "Model", "Mean Value"]
#
#     # Initialize rows for the table
#     rows = []
#
#     # Loop through each metric
#     for metric in metrics:
#         for model_name, scores in models.items():
#             mean_value = mean(scores[metric])
#             rows.append([metric, model_name, f"{mean_value:.3f}"])
#
#     # Print the table using tabulate
#     print(tabulate(rows, headers=header, tablefmt="grid"))
#
# # Call the function with the example metrics
# gather_and_print_metrics(metrics)
#
# def translate_name(name):
#     if name == "kNN":
#         return 1
#     elif name == "NB":
#         return 2
#     elif name == "RL":
#         return 3
#     elif name == "SVC":
#         return 4
#     return -1
#
# for i, (model_a_name, model_a_scores) in enumerate(models.items()):
#     for j, (model_b_name, model_b_scores) in enumerate(models.items()):
#         if i < j:
#             print(f"\nComparing {model_a_name} and {model_b_name}")
#             for metric in metrics:
#                 perform_test(model_a_scores[metric], model_b_scores[metric], metric, translate_name(model_a_name), translate_name(model_b_name))

from statistical_Test import perform_test
from utils.ResultHelper import get_values, ResultHelper
from tabulate import tabulate
from numpy import mean

# =====================
# Configuration
# =====================
dataset_a = "cnn"
dataset_b = "smote"
metrics = ['BAC']

rh = ResultHelper()

# =====================
# Load results
# =====================
models_a = {
    "kNN": get_values(f"../results/e1/{dataset_a}/kNN_{dataset_a}.csv"),
    "NB":  get_values(f"../results/e1/{dataset_a}/NB_{dataset_a}.csv"),
    "RL":  get_values(f"../results/e1/{dataset_a}/LR_{dataset_a}.csv"),
    "SVC": get_values(f"../results/e1/{dataset_a}/SVC_{dataset_a}.csv"),
}

models_b = {
    "kNN": get_values(f"../results/e1/{dataset_b}/kNN_{dataset_b}.csv"),
    "NB":  get_values(f"../results/e1/{dataset_b}/NB_{dataset_b}.csv"),
    "RL":  get_values(f"../results/e1/{dataset_b}/LR_{dataset_b}.csv"),
    "SVC": get_values(f"../results/e1/{dataset_b}/SVC_{dataset_b}.csv"),
}

# =====================
# Radar plot (optional)
# =====================
rh.plot_radar_combined(
    {
        f"{model}_{dataset_a}": scores
        for model, scores in models_a.items()
    } | {
        f"{model}_{dataset_b}": scores
        for model, scores in models_b.items()
    },
    "./"
)

# =====================
# Mean metrics table
# =====================
def gather_and_print_metrics():
    header = ["Dataset", "Model", "Metric", "Mean Value"]
    rows = []

    for dataset_name, models in [(dataset_a, models_a), (dataset_b, models_b)]:
        for model_name, scores in models.items():
            for metric in metrics:
                rows.append([
                    dataset_name,
                    model_name,
                    metric,
                    f"{mean(scores[metric]):.3f}"
                ])

    print(tabulate(rows, headers=header, tablefmt="grid"))

gather_and_print_metrics()

# =====================
# Statistical tests
# =====================
for model_name in models_a.keys():
    print(f"\nComparing {model_name}: {dataset_a} vs {dataset_b}")
    for metric in metrics:
        perform_test(
            models_a[model_name][metric],
            models_b[model_name][metric],
            metric,
            dataset_a,
            dataset_b
        )