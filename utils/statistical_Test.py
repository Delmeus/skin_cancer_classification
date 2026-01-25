from scipy.stats import ttest_rel
from utils.ResultHelper import get_values
# Accuracy values for each fold
# acc_knn = [0.7968047928107839,0.8002995506739891,0.8017973040439341,0.799800299550674,0.8112830753869196,0.8077883175237144,0.8017973040439341,0.8047928107838243,0.8117823265102346,0.7973040439340988]  # Replace with actual fold values
# acc_nb = [0.7688467299051422,0.7493759360958562,0.7663504742885672,0.7538691962056915,0.7668497254118821,0.7558662006989516,0.7533699450823764,0.7693459810284573,0.7668497254118821,0.7538691962056915]   # Replace with actual fold values

def perform_test(a_scores, b_scores, metric_name, a_name="A", b_name="B"):
    print(f"For {metric_name} ", end="")
    t_stat, p_value = ttest_rel(a_scores, b_scores)
    # print(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        # print(f"Significant difference in {metric_name} between {a_name} and {b_name}.")
        # Determine which performs better
        mean_a = sum(a_scores) / len(a_scores)
        mean_b = sum(b_scores) / len(b_scores)
        if mean_a > mean_b:
            print(f"{a_name} performs significantly better than {b_name}.")
            return 1
        else:
            print(f"{b_name} performs significantly better than {a_name}.")
            return 2
    else:
        print(f"there is no significant difference.")
        return 3
#
# metrics = ['Acc', 'Rec', 'Prec', 'Spec', 'F1', 'BAC', 'G-mean']
# knn = get_values("results/e1/hog/imbalanced/kNN.csv")
# nb = get_values("results/e1/hog/imbalanced/Naiwny klasyfikator Bayesowski.csv")
# svm = get_values("results/e1/hog/imbalanced/SVM.csv")
# rl = get_values("results/e1/hog/imbalanced/Regresja logistyczna.csv")
#
# print("-------------TEST 1-------------")
# for metric in metrics:
#     print(f"\nComparing {metric}")
#     perform_test(knn[metric], nb[metric], metric, "kNN", "NB")
#
# print("-------------TEST 2-------------")
# for metric in metrics:
#     print(f"\nComparing {metric}")
#     perform_test(knn[metric], rl[metric], metric, "kNN", "RL")
#
# print("-------------TEST 3-------------")
# for metric in metrics:
#     print(f"\nComparing {metric}")
#     perform_test(knn[metric], svm[metric], metric, "kNN", "SVM")