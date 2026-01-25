import os
import pandas as pd
import numpy as np
from scipy import stats
import itertools

# Configuration
# Assuming script is run from project root, so results are in results/e1
RESULTS_DIR = os.path.join("results", "e1")

# Mapping of folder_name -> filename_suffix
METHODS = {
    'imb': 'imbalanced',
    'cnn': 'cnn',
    'ros': 'ros',
    'rus': 'rus',
    'smote': 'smote'
}

CLASSIFIERS = ['kNN', 'LR', 'NB', 'SVC']
METRIC = 'BAC'

def get_metric_scores(method_dir, classifier, suffix, metric_name):
    """
    Reads the CSV file for a given method and classifier, extracting the specific metric row.
    """
    filename = f"{classifier}_{suffix}.csv"
    filepath = os.path.join(RESULTS_DIR, method_dir, filename)
    
    if not os.path.exists(filepath):
        # Fallback check for different naming if necessary?
        # For now, strictly follow the plan.
        print(f"  [Warn] File not found: {filepath}")
        return None

    try:
        # Read CSV. The file format observed: MetricName, val1, val2, ...
        # No header row. First column is index.
        df = pd.read_csv(filepath, header=None, index_col=0)
        
        if metric_name in df.index:
            # Return values as numpy array
            return df.loc[metric_name].values.astype(float)
        else:
            print(f"  [Warn] Metric '{metric_name}' not found in {filepath}")
            return None
    except Exception as e:
        print(f"  [Error] Failed to read {filepath}: {e}")
        return None

def main():
    print(f"Starting Statistical Analysis for metric: {METRIC}")
    print(f"Looking in: {os.path.abspath(RESULTS_DIR)}\n")

    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory '{RESULTS_DIR}' does not exist.")
        return

    for clf in CLASSIFIERS:
        print("="*60)
        print(f"Classifier: {clf}")
        print("-" * 60)
        
        clf_data = {}
        
        # 1. Collect Data
        for method_dir, suffix in METHODS.items():
            scores = get_metric_scores(method_dir, clf, suffix, METRIC)
            if scores is not None:
                clf_data[method_dir] = scores
        
        if not clf_data:
            print(f"No data found for classifier {clf}. Skipping.")
            continue
            
        # 2. Align Data (Ensure same number of folds/samples)
        lengths = [len(v) for v in clf_data.values()]
        if len(set(lengths)) > 1:
            min_len = min(lengths)
            print(f"  [Info] Data length mismatch {lengths}. Truncating to {min_len} samples.")
            for k in clf_data:
                clf_data[k] = clf_data[k][:min_len]
        
        df_scores = pd.DataFrame(clf_data)
        
        # Print summary stats
        print("\nMean Scores (Higher is better):")
        print(df_scores.mean().sort_values(ascending=False))
        
        # 3. Friedman Test
        # Need at least 3 methods for Friedman usually, or 2 (equivalent to sign test?)
        if len(df_scores.columns) < 2:
            print("Not enough methods to perform statistical tests.")
            continue

        samples = [df_scores[col] for col in df_scores.columns]
        stat, p_value = stats.friedmanchisquare(*samples)
        
        print(f"\nFriedman Test:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  p-value:   {p_value:.6f}")
        
        # 4. Post-hoc Analysis (if significant)
        alpha = 0.05
        if p_value < alpha:
            print("\nResult is significant (p < 0.05). Performing pairwise Wilcoxon signed-rank tests:")
            
            method_names = list(df_scores.columns)
            pairs = list(itertools.combinations(method_names, 2))
            
            # Simple pairwise comparison
            significant_pairs = []
            print(f"  {'Pair':<20} | {'p-value':<10} | {'Significant?'}")
            print(f"  {'-'*20} | {'-'*10} | {'-'*12}")
            
            for m1, m2 in pairs:
                try:
                    s_stat, p_w = stats.wilcoxon(df_scores[m1], df_scores[m2])
                    is_sig = "Yes" if p_w < alpha else "No"
                    print(f"  {m1} vs {m2:<14} | {p_w:.6f}   | {is_sig}")
                    if is_sig == "Yes":
                        significant_pairs.append((m1, m2))
                except ValueError as e:
                    # Happens if all differences are zero
                    print(f"  {m1} vs {m2:<14} | N/A (identical scores?)")
            
        else:
            print("\nResult is NOT significant. No post-hoc tests needed.")
            
        print("\n")

if __name__ == "__main__":
    main()
