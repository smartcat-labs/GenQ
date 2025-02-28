import datetime
import warnings
from typing import List
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from loguru import logger 
from pathlib import Path
from modules.utils import load_config

"""
result_analysis.py

This script performs analysis on generated_results.csv,
creates several plots comparing ROUGE scores, text length groups,
and semantic similarity measures, and saves all plots into a folder
named with the current date and time.

Run the script with:
    python -m modules.eval.results_analysis config/analysis_config.yaml
"""

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = Path(f"modules/eval/result_analysis/{dt}")
output_path.mkdir(parents=True, exist_ok=True)

logger.add(f"{output_path}/result_analysis.log", level="INFO")

def plot_quick_comparison(data: pd.DataFrame, score_columns: List[str], model_names: List[str]) -> None:
    """
    Density plot of ROUGE scores for multiple models in the same graph.

    Parameters:
        data (pd.DataFrame): The dataset containing ROUGE scores.
        score_columns (List[str]): List of column names for ROUGE scores.
        model_names (List[str]): List of model names to compare.

    Returns:
        None: Displays and saves the plots.
    """
    if len(model_names) != 2:
        raise ValueError("This function currently supports exactly two models for comparison.")

    # Extract model names
    model_1, model_2 = model_names

    # Filter data for each model
    model_1_data = data[data['model'] == model_1]
    model_2_data = data[data['model'] == model_2]

    # Set up the overall style and aesthetics
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(18, 12))

    # Iterate over each ROUGE score column to create subplots
    for i, score in enumerate(score_columns, 1):
        plt.subplot(2, 2, i)
        sns.kdeplot(model_1_data[score], fill=True, color='blue', alpha=0.5, label=model_1, linewidth=2)
        sns.kdeplot(model_2_data[score], fill=True, color='orange', alpha=0.5, label=model_2, linewidth=2)
        plt.title(f'Distribution of {score}', fontsize=14, fontweight='bold')
        plt.xlabel(f'{score} (%)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10, title="Model", title_fontsize=12, loc='upper right')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.suptitle(f'Comparison of ROUGE Score Distributions: {model_1} vs {model_2}', fontsize=16, fontweight='bold', y=1.02)

    # Save and display the plot

    plt.savefig(f"{output_path}/density_comparison.png", dpi=300, bbox_inches="tight")


def plot_detailed_comparison(data: pd.DataFrame, score_columns: List[str], model_names: List[str]) -> None:
    """
    Plots separate histograms of ROUGE scores for two models side by side.

    Parameters:
        data (pd.DataFrame): The dataset containing ROUGE scores.
        score_columns (List[str]): List of column names for ROUGE scores.
        model_names (List[str]): List of model names to compare (must contain exactly two models).

    Returns:
        None: Displays and saves the plots.
    """
    if len(model_names) != 2:
        raise ValueError("This function currently supports exactly two models for comparison.")

    # Extract model names
    model_1, model_2 = model_names

    # Filter data for each model
    model_1_data = data[data['model'] == model_1]
    model_2_data = data[data['model'] == model_2]

    # Set up the overall style and aesthetics
    sns.set_theme(style="whitegrid")

    # Create subplots: One row per ROUGE score, Two columns (one for each model)
    fig, axes = plt.subplots(nrows=len(score_columns), ncols=2, figsize=(14, 4 * len(score_columns)))
    
    for i, score in enumerate(score_columns):
        # Plot model 1 on the left side
        sns.histplot(model_1_data[score], kde=True, bins=20, color='blue', edgecolor='black', ax=axes[i, 0])
        axes[i, 0].set_title(f'{model_1} - {score}', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel(f'{score} (%)', fontsize=12)
        axes[i, 0].set_ylabel('Frequency', fontsize=12)

        # Plot model 2 on the right side
        sns.histplot(model_2_data[score], kde=True, bins=20, color='orange', edgecolor='black', ax=axes[i, 1])
        axes[i, 1].set_title(f'{model_2} - {score}', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel(f'{score} (%)', fontsize=12)
        axes[i, 1].set_ylabel('Frequency', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.suptitle(f'Comparison of ROUGE Score Histograms: {model_1} vs {model_2}', fontsize=16, fontweight='bold', y=1.02)

    # Save and display the plot
    plt.savefig(f"{output_path}/histogram_comparison.png", dpi=300, bbox_inches="tight")

def get_results_for_zero_scores(data: pd.DataFrame, model_name: str,
                                save: bool = False) -> pd.DataFrame:
    """
    Retrieves results where all ROUGE scores are zero for a given model.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores.
        model_name (str): Name of the model.
        save (bool): Whether to save the results to a CSV file.

    Returns:
        pd.DataFrame: DataFrame with zero scores.
    """
    zero_scores_df = data[
        (data['rouge1'] == 0) &
        (data['rouge2'] == 0) &
        (data['rougeLsum'] == 0) &
        (data['rougeL'] == 0) &
        (data['model'] == model_name)
    ]

    if save:
        zero_scores_df.to_csv(f"{output_path}/zeros.csv", index=False)

    return zero_scores_df


def get_outperformed_results(data: pd.DataFrame, first_model: str, second_model: str,
                             save: bool = False) -> pd.DataFrame:
    """
    Merges results from two models and returns rows where the first model outperforms the second.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores and outputs.
        first_model (str): Name of the first model.
        second_model (str): Name of the second model.
        save (bool): Whether to save the results to a CSV file.

    Returns:
        pd.DataFrame: DataFrame with better results for the first model.
    """
    df_1 = data[data['model'] == first_model]
    df_2 = data[data['model'] == second_model]

    merged_df = pd.merge(
        df_1[['target_query', 'generated_output', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']],
        df_2[['target_query', 'generated_output', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']],
        on='target_query',
        suffixes=(f'_{first_model}', f'_{second_model}')
    )

    better_results = merged_df[
        (merged_df[f'rouge1_{first_model}'] > merged_df[f'rouge1_{second_model}']) &
        (merged_df[f'rouge2_{first_model}'] > merged_df[f'rouge2_{second_model}']) &
        (merged_df[f'rougeL_{first_model}'] > merged_df[f'rougeL_{second_model}']) &
        (merged_df[f'rougeLsum_{first_model}'] > merged_df[f'rougeLsum_{second_model}'])
    ]

    filtered_df = better_results[[
        "target_query",
        f'generated_output_{first_model}',
        f'generated_output_{second_model}',
    ]]

    if save:
        filtered_df.to_csv(f"{output_path}/better_{first_model}.csv", index=False)

    return filtered_df


def get_best_results(data: pd.DataFrame, score: float, model_name: str,
                     save: bool = False) -> pd.DataFrame:
    """
    Retrieves results where the ROUGE-L score is greater than or equal to a threshold.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores.
        score (float): Threshold for ROUGE-L score.
        model_name (str): Name of the model.
        save (bool): Whether to save the results to a CSV file.

    Returns:
        pd.DataFrame: DataFrame with best results.
    """
    best = data[(data['rougeL'] >= score) & (data['model'] == model_name)]

    if save:
        best.to_csv(f"{output_path}/best.csv", index=False)

    return best


def get_worst_results(data: pd.DataFrame, model_name: str, score: float = 0,
                      save: bool = False) -> pd.DataFrame:
    """
    Retrieves results where the ROUGE-L score is less than or equal to a threshold.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores.
        model_name (str): Name of the model.
        score (float): Threshold for ROUGE-L score.
        save (bool): Whether to save the results to a CSV file.

    Returns:
        pd.DataFrame: DataFrame with worst results.
    """
    worst = data[(data['rougeL'] <= score) & (data['model'] == model_name)]

    if save:
        worst.to_csv(f"{output_path}/worst.csv", index=False)

    return worst


def calculate_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates average ROUGE scores grouped by generated output size.

    Parameters:
        data (pd.DataFrame): DataFrame with columns 'generated_size', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'.

    Returns:
        pd.DataFrame: DataFrame with average scores.
    """
    averages = data.groupby('generated_size')[['rouge1', 'rouge2', 'rougeL', 'rougeLsum']].mean().reset_index()
    averages.columns = ['size', 'rouge1_avg', 'rouge2_avg', 'rougeL_avg', 'rougeLsum_avg']
    return averages


def plot_by_group_sizes(df: pd.DataFrame) -> None:
    """
    Creates bar plots with line overlays showing average ROUGE scores and differences by generated size.

    Parameters:
        df (pd.DataFrame): DataFrame with average ROUGE scores by group size.
    """
    score_columns = ["rouge1_avg", "rouge2_avg", "rougeL_avg", "rougeLsum_avg"]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes = axes.flatten()

    colors = [
        ('#1f77b4', '#ff7f0e'),  # rouge1_avg: Blue for bars, Orange for line
        ('#2ca02c', '#d62728'),  # rouge2_avg: Green for bars, Red for line
        ('#9467bd', '#8c564b'),  # rougeL_avg: Purple for bars, Brown for line
        ('#e377c2', '#7f7f7f')   # rougeLsum_avg: Pink for bars, Gray for line
    ]

    for i, score in enumerate(score_columns):
        ranked_sizes = df[["size", score]].copy()
        ranked_sizes["score_diff"] = ranked_sizes[score].diff().fillna(0)
        bar_color, line_color = colors[i]

        axes[i].bar(ranked_sizes["size"].astype(str), ranked_sizes[score],
                    label="Score (%)", color=bar_color, alpha=0.6)
        axes[i].plot(ranked_sizes["size"].astype(str), ranked_sizes["score_diff"],
                     label="Score Difference", color=line_color, marker='o')
        axes[i].set_title(f"{score.replace('_', ' ').upper()} Rankings and Score Differences")
        axes[i].set_xlabel("Size (Words)")
        axes[i].set_ylabel("Score (%)")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/group_sizes.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_cosine_similarities(df: pd.DataFrame, model_ST: SentenceTransformer) -> List[float]:
    """
    Computes cosine similarity between target queries and generated outputs.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'target_query' and 'generated_output' columns.
        model_ST (SentenceTransformer): Pretrained Sentence Transformer model.

    Returns:
        List[float]: List of cosine similarity scores.
    """
    logger.info("Computing cosine similarities for dataset with {} rows", len(df))
    cosine_similarities = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Cosine Similarities"):
        target = str(row["target_query"])
        generated = str(row["generated_output"])
        embeddings = model_ST.encode([target, generated], convert_to_tensor=True)
        cos_sim = util.cos_sim(embeddings[0], embeddings[1])
        cosine_similarities.append(float(cos_sim.item()))
    return cosine_similarities


def plot_similarity_vs_rouge(filtered_df: pd.DataFrame) -> None:
    """
    Creates scatter plots showing the relationship between semantic similarity and ROUGE scores,
    then saves the combined figure.

    Parameters:
        filtered_df (pd.DataFrame): DataFrame containing 'semantic_similarity' and ROUGE score columns.
    """
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    colors = ["blue", "red", "green", "purple"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        sns.scatterplot(data=filtered_df, x="semantic_similarity", y=metric,
                        color=color, alpha=0.7, ax=axes[i])
        axes[i].set_title(f"Semantic Similarity vs. {metric.upper()}", fontsize=12)
        axes[i].set_xlabel("Semantic Similarity (Cosine Similarity)")
        axes[i].set_ylabel(f"{metric.upper()} Score")
        axes[i].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{output_path}/similarity_vs_rouge.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_semantic_similarity_distribution(filtered_df: pd.DataFrame) -> None:
    """
    Creates a histogram of semantic similarity scores and saves the plot.

    Parameters:
        filtered_df (pd.DataFrame): DataFrame containing the 'semantic_similarity' column.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(filtered_df["semantic_similarity"], bins=25, kde=True, color="blue")
    plt.title("Distribution of Semantic Similarity Scores", fontsize=14)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"{output_path}/semantic_similarity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_average_scores_by_model(data: pd.DataFrame, score_columns: List[str]) -> None:
    """
    Plots the average ROUGE scores for each model.

    Parameters:
        data (pd.DataFrame): The dataset containing ROUGE scores and model names.
        score_columns (List[str]): List of column names for ROUGE scores.
    """
    avg_scores = data.groupby('model')[score_columns].mean().reset_index()
    
    plt.figure(figsize=(15, 12))
    sns.set_theme(style="whitegrid")
    
    for i, score in enumerate(score_columns, 1):
        plt.subplot(2, 2, i)
        ax = sns.barplot(data=avg_scores, x='model', y=score, palette='tab10')
        plt.title(f'{score.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(f'Average {score.upper()} (%)', fontsize=12)
        
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Average ROUGE Scores by Model', fontsize=16, fontweight='bold', y=1.02)
    
    # Save and display the plot
    plt.savefig(f"{output_path}/average_scores_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    logger.info(f"Results will be saved in folder: {output_path}")
    # Load the generated results
    parser = argparse.ArgumentParser(description="Run evaluation with a config file.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config, type="analysis")
    results = pd.read_csv(config["results_path"])

    logger.info("Loading results from: {}", config["results_path"])

    models = results["model"].unique()
    logger.info("Loaded dataset with {} rows and {} columns", results.shape[0], results.shape[1])

    if len(models) < 2:
        logger.error("Error: Not enough models found in the dataset.")
        return
    
    if len(models) > 2:
        logger.warning("There are more than two models. Using two models set in config.")
        first_model, second_model = models[config["compare_models"][0]], models[config["compare_models"][1]]

    if len(models) == 2:
        first_model, second_model = models
        
    logger.info(f"Comparing models: {first_model} vs {second_model}")

    score_columns = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

    # Create and save the comparison plots
    plot_detailed_comparison(results, score_columns, [first_model, second_model])
    plot_quick_comparison(results, score_columns, [first_model, second_model])
    plot_average_scores_by_model(results, score_columns)

    # Retrieve and print various results
    get_outperformed_results(results, second_model, first_model, save=config["save_outperformed"])
    get_results_for_zero_scores(results, first_model, save= config["save_zeros"])
    get_best_results(data=results, score=50, model_name=first_model, save=config["save_best"])
    get_worst_results(data=results, model_name=first_model, save=config["save_worst"])

    # Analyze generated query lengths
    fine_tuned_df = results[results["model"] == first_model]
    max_length = fine_tuned_df["generated_output"].apply(lambda x: len(x.split())).max()
    min_length = fine_tuned_df["generated_output"].apply(lambda x: len(x.split())).min()
    logger.info(f"Longest generated query has {max_length} words.")
    logger.info(f"Shortest generated query has {min_length} words.")

    # Categorize generated outputs by word count and calculate averages
    fine_tuned_df["generated_size"] = fine_tuned_df["generated_output"].apply(lambda x: len(x.split()))
    averages = calculate_averages(fine_tuned_df[["generated_size", "rouge1", "rouge2", "rougeL", "rougeLsum"]])
    logger.info("Averages by generated output size:")
    logger.info(averages)
    plot_by_group_sizes(averages)
    
    if config["compute_similarity"]:
        # Compute semantic similarities and plot the results
        model_ST = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Computing semantic similarity scores...")
        results["semantic_similarity"] = compute_cosine_similarities(results, model_ST)
        filtered_df = results[results["model"] == first_model]
        plot_similarity_vs_rouge(filtered_df)
        plot_semantic_similarity_distribution(filtered_df)

    logger.info("Finished analysis.")


if __name__ == "__main__":
    main()
