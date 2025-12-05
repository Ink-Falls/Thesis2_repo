import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURATION ---
RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "hyperparameter_search_log.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "defense_graphs_hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD EVIDENCE
print(f"--- Loading Evidence from: {LOG_FILE} ---")
df = pd.read_csv(LOG_FILE)
target_metric = "mean_test_score"

# SET STYLE
sns.set_theme(style="whitegrid")  # Clean scientific look

models = df["model"].unique()

for model in models:
    print(f"Processing Model: {model}...")
    model_df = df[df["model"] == model].copy()

    # Identify params
    metadata_cols = [
        "dataset",
        "model",
        "run_id",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    params = [c for c in model_df.columns if c not in metadata_cols]
    params = [p for p in params if model_df[p].notna().any()]  # Filter empty

    num_params = len(params)
    if num_params == 0:
        continue

    # Dynamic Plot Layout
    cols = 3
    rows = (num_params // cols) + (1 if num_params % cols > 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if num_params > 1 else [axes]

    for i, param in enumerate(params):
        ax = axes[i]

        try:
            # 1. SANITIZE DATA
            plot_data = model_df.copy()

            # Check if Numeric
            is_numeric = pd.api.types.is_numeric_dtype(plot_data[param])

            if not is_numeric:
                # Fill NaNs for categorical
                plot_data[param] = plot_data[param].fillna("None").astype(str)

            # 2. PLOT THE "RIGOR" (The Raw Data Dots)
            # We use 'stripplot' or 'scatterplot' to show the individual runs
            # Alpha makes them transparent so they don't block the line
            if is_numeric:
                sns.scatterplot(
                    data=plot_data,
                    x=param,
                    y=target_metric,
                    hue="dataset",
                    alpha=0.3,
                    ax=ax,
                    legend=False,
                    palette="viridis",
                )
            else:
                sns.stripplot(
                    data=plot_data,
                    x=param,
                    y=target_metric,
                    hue="dataset",
                    alpha=0.4,
                    dodge=True,
                    ax=ax,
                    legend=False,
                    palette="viridis",
                    jitter=True,
                )

            # 3. PLOT THE "IMPROVEMENT" (The Trend Line)
            # This draws the Mean and the Confidence Interval
            if is_numeric:
                sns.lineplot(
                    data=plot_data,
                    x=param,
                    y=target_metric,
                    hue="dataset",
                    style="dataset",
                    markers=True,
                    dashes=False,  # Add markers to line
                    ax=ax,
                    palette="viridis",
                    linewidth=2.5,
                    errorbar=("ci", 95),
                )
            else:
                # For categorical, we use pointplot to connect the means
                sns.pointplot(
                    data=plot_data,
                    x=param,
                    y=target_metric,
                    hue="dataset",
                    dodge=0.4,
                    join=False,
                    capsize=0.1,  # Don't connect lines for categories, just show mean point
                    ax=ax,
                    palette="viridis",
                    errorbar=("ci", 95),
                )

            # 4. FORMATTING
            ax.set_title(f"Impact of {param}", fontsize=12, fontweight="bold")
            ax.set_ylabel("F1 Score")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Fix Legend: Only show it on the last plot to save space
            if i == num_params - 1:
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            else:
                if ax.get_legend():
                    ax.get_legend().remove()

        except Exception as e:
            print(f"   [WARNING] Could not plot param '{param}': {e}")
            ax.text(0.5, 0.5, "Data Error", ha="center", va="center")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Hyperparameter Evolution: {model}\n(Lines = Mean Trend | Dots = Individual Runs)",
        fontsize=16,
    )
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"Hybrid_{model}.png")
    plt.savefig(output_path, dpi=300)
    print(f"   -> Saved Hybrid Graph: {output_path}")
    plt.close()

print("--- Hybrid Visualization Protocol Complete ---")
