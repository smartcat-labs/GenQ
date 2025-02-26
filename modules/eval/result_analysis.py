#!/usr/bin/env python3
"""
result_analysis.py

This script performs analysis on generated_results.csv,
creates several plots comparing ROUGE scores, text length groups,
and semantic similarity measures, and saves all plots into a folder
named with the current date and time.
"""

import os
import datetime
import warnings
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from loguru import logger 

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("eval/result_analysis/", dt)
os.makedirs(output_dir, exist_ok=True)
print(f"Plots and outputs will be saved in the folder: {output_dir}")
logger.add(f"{output_dir}result_analysis.log", rotation="1 MB", level="INFO")

def plot_quick_comparison(data: pd.DataFrame, score_columns: List[str],
                          fine_tuned_model_name: str, original_model_name: str,
                          save_dir: str) -> None:
    """
    Creates density plots of ROUGE scores for two models and saves the figure.

    Parameters:
        data (pd.DataFrame): DataFrame containing ROUGE scores.
        score_columns (List[str]): List of ROUGE score column names.
        fine_tuned_model_name (str): Name of the fine-tuned model.
        original_model_name (str): Name of the original model.
        save_dir (str): Directory to save the plot.
    """
    # Filter data for each model
    fine_tuned_data = data[data['model'] == fine_tuned_model_name]
    original_data = data[data['model'] == original_model_name]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 12))

    # Create subplots for each ROUGE score
    for i, score in enumerate(score_columns, 1):
        plt.subplot(2, 2, i)
        sns.kdeplot(fine_tuned_data[score], fill=True, color='blue', alpha=0.5,
                    label='Fine-Tuned Model', linewidth=2)
        sns.kdeplot(original_data[score], fill=True, color='orange', alpha=0.5,
                    label='Original Model', linewidth=2)
        plt.title(f'Distribution of {score}', fontsize=14, fontweight='bold')
        plt.xlabel(f'{score} (%)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10, title="Model", title_fontsize=12, loc='upper right')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.suptitle('Comparison of ROUGE Score Distributions', fontsize=16, fontweight='bold', y=1.02)
    output_path = os.path.join(save_dir, "quick_comparison.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_detailed_comparison(data: pd.DataFrame, score_columns: List[str],
                             fine_tuned_model_name: str, original_model_name: str,
                             save_dir: str) -> None:
    """
    Creates histograms (with KDE) of ROUGE scores for each model and saves the figure.

    Parameters:
        data (pd.DataFrame): DataFrame containing ROUGE scores.
        score_columns (List[str]): List of ROUGE score column names.
        fine_tuned_model_name (str): Name of the fine-tuned model.
        original_model_name (str): Name of the original model.
        save_dir (str): Directory to save the plot.
    """
    fine_tuned_data = data[data['model'] == fine_tuned_model_name]
    original_data = data[data['model'] == original_model_name]

    plt.figure(figsize=(15, 20))
    for i, score in enumerate(score_columns, 1):
        # Plot for fine-tuned model
        plt.subplot(4, 2, 2 * i - 1)
        sns.histplot(fine_tuned_data[score], kde=True, bins=20, color='blue', edgecolor='black')
        plt.title(f'Distribution of {score} for {fine_tuned_model_name}')
        plt.xlabel(f'{score} (%)')
        plt.ylabel('Frequency')

        # Plot for original model
        plt.subplot(4, 2, 2 * i)
        sns.histplot(original_data[score], kde=True, bins=20, color='orange', edgecolor='black')
        plt.title(f'Distribution of {score} for {original_model_name}')
        plt.xlabel(f'{score} (%)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    output_path = os.path.join(save_dir, "detailed_comparison.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_results_for_zero_scores(data: pd.DataFrame, model_name: str,
                                save: bool = False, display_head: bool = False) -> pd.DataFrame:
    """
    Retrieves results where all ROUGE scores are zero for a given model.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores.
        model_name (str): Name of the model.
        save (bool): Whether to save the results to a CSV file.
        display_head (bool): Whether to print the head of the DataFrame.

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
        zero_scores_df.to_csv("zeros.csv", index=False)

    if display_head:
        print("Zero Scores DataFrame Head:")
        print(zero_scores_df.head())

    return zero_scores_df


def get_outperformed_results(data: pd.DataFrame, first_model: str, second_model: str,
                             save: bool = False, display_head: bool = False) -> pd.DataFrame:
    """
    Merges results from two models and returns rows where the first model outperforms the second.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores and outputs.
        first_model (str): Name of the first model.
        second_model (str): Name of the second model.
        save (bool): Whether to save the results to a CSV file.
        display_head (bool): Whether to print the head of the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with better results for the first model.
    """
    df_1 = data[data['model'] == first_model]
    df_2 = data[data['model'] == second_model]

    merged_df = pd.merge(
        df_1[['title', 'target_query', 'generated_output', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']],
        df_2[['title', 'target_query', 'generated_output', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']],
        on='title',
        suffixes=(f'_{first_model}', f'_{second_model}')
    )

    better_results = merged_df[
        (merged_df[f'rouge1_{first_model}'] > merged_df[f'rouge1_{second_model}']) &
        (merged_df[f'rouge2_{first_model}'] > merged_df[f'rouge2_{second_model}']) &
        (merged_df[f'rougeL_{first_model}'] > merged_df[f'rougeL_{second_model}']) &
        (merged_df[f'rougeLsum_{first_model}'] > merged_df[f'rougeLsum_{second_model}'])
    ]

    filtered_df = better_results[[
        'title',
        f'target_query_{second_model}',
        f'generated_output_{first_model}',
        f'generated_output_{second_model}',
    ]]

    if save:
        filtered_df.to_csv(f"better_{first_model}.csv", index=False)

    if display_head:
        print("Outperformed Results DataFrame Head:")
        print(filtered_df.head())

    return filtered_df


def get_best_results(data: pd.DataFrame, score: float, model_name: str,
                     save: bool = False, display_head: bool = False) -> pd.DataFrame:
    """
    Retrieves results where the ROUGE-L score is greater than or equal to a threshold.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores.
        score (float): Threshold for ROUGE-L score.
        model_name (str): Name of the model.
        save (bool): Whether to save the results to a CSV file.
        display_head (bool): Whether to print the head of the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with best results.
    """
    best = data[(data['rougeL'] >= score) & (data['model'] == model_name)]

    if save:
        best.to_csv("best50.csv", index=False)

    if display_head:
        print("Best Results DataFrame Head:")
        print(best.head())

    return best


def get_worst_results(data: pd.DataFrame, model_name: str, score: float = 0,
                      save: bool = False, display_head: bool = False) -> pd.DataFrame:
    """
    Retrieves results where the ROUGE-L score is less than or equal to a threshold.

    Parameters:
        data (pd.DataFrame): DataFrame with ROUGE scores.
        model_name (str): Name of the model.
        score (float): Threshold for ROUGE-L score.
        save (bool): Whether to save the results to a CSV file.
        display_head (bool): Whether to print the head of the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with worst results.
    """
    worst = data[(data['rougeL'] <= score) & (data['model'] == model_name)]

    if save:
        worst.to_csv("worst50.csv", index=False)

    if display_head:
        print("Worst Results DataFrame Head:")
        print(worst.head())

    return worst


def categorize_generated_output(data: pd.DataFrame, bins: List[float],
                                labels: List[str]) -> pd.DataFrame:
    """
    Categorizes generated outputs based on word count.

    Parameters:
        data (pd.DataFrame): DataFrame containing a 'generated_output' column.
        bins (List[float]): List of bin edges.
        labels (List[str]): Labels for each bin.

    Returns:
        pd.DataFrame: DataFrame with a new 'generated_size' column.
    """
    if len(labels) != len(bins) - 1:
        raise ValueError("Number of labels must be equal to the number of bins minus 1.")

    data["generated_size"] = pd.cut(
        data["generated_output"].apply(lambda x: len(x.split())),
        bins=bins,
        labels=labels,
        right=False
    )

    return data


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


def plot_by_group_sizes(df: pd.DataFrame, save_dir: str) -> None:
    """
    Creates bar plots with line overlays showing average ROUGE scores and differences by generated size.

    Parameters:
        df (pd.DataFrame): DataFrame with average ROUGE scores by group size.
        save_dir (str): Directory to save the plot.
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
    output_path = os.path.join(save_dir, "group_sizes.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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


def plot_similarity_vs_rouge(filtered_df: pd.DataFrame, save_dir: str) -> None:
    """
    Creates scatter plots showing the relationship between semantic similarity and ROUGE scores,
    then saves the combined figure.

    Parameters:
        filtered_df (pd.DataFrame): DataFrame containing 'semantic_similarity' and ROUGE score columns.
        save_dir (str): Directory to save the plot.
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
    output_path = os.path.join(save_dir, "similarity_vs_rouge.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_semantic_similarity_distribution(filtered_df: pd.DataFrame, save_dir: str) -> None:
    """
    Creates a histogram of semantic similarity scores and saves the plot.

    Parameters:
        filtered_df (pd.DataFrame): DataFrame containing the 'semantic_similarity' column.
        save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(filtered_df["semantic_similarity"], bins=25, kde=True, color="blue")
    plt.title("Distribution of Semantic Similarity Scores", fontsize=14)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    output_path = os.path.join(save_dir, "semantic_similarity_distribution.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Create an output folder named with the current date and time (YYYYMMDD_HHMMSS)
    # dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join("eval/result_analysis/", dt)
    # os.makedirs(output_dir, exist_ok=True)
    # print(f"Plots and outputs will be saved in the folder: {output_dir}")
    logger.info("Results will be saved in folder: {}", output_dir)
    # Load the generated results
    results = pd.read_csv("/home/petar/Documents/trainings/14-02/generated_results.csv")
    logger.info("Loading results from: {}", results)
    models = results["model"].unique()
    logger.info("Loaded dataset with {} rows and {} columns", results.shape[0], results.shape[1])

    if len(models) < 2:
        print("Error: Not enough models found in the dataset.")
        return

    first_model = models[0]
    second_model = models[1]
    logger.info("Comparing models: {} vs {}", first_model, second_model)

    score_columns = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

    # Create and save the comparison plots
    plot_detailed_comparison(results, score_columns, first_model, second_model, output_dir)
    plot_quick_comparison(results, score_columns, first_model, second_model, output_dir)

    # Retrieve and print various results
    get_outperformed_results(results, second_model, first_model, display_head=True)
    get_results_for_zero_scores(results, first_model, display_head=True)
    get_best_results(data=results, score=50, model_name=first_model, display_head=True)
    get_worst_results(data=results, model_name=first_model, display_head=True)

    # Analyze generated query lengths
    fine_tuned_df = results[results["model"] == first_model]
    max_length = fine_tuned_df["generated_output"].apply(lambda x: len(x.split())).max()
    min_length = fine_tuned_df["generated_output"].apply(lambda x: len(x.split())).min()
    print(f"Longest generated query has {max_length} words.")
    print(f"Shortest generated query has {min_length} words.")

    # Categorize generated outputs by word count and calculate averages
    bins = [0, 3, 5, 7, float("inf")]
    labels = ["1-2", "3-4", "5-6", "7+"]
    fine_tuned_df = categorize_generated_output(fine_tuned_df, bins, labels)
    averages = calculate_averages(fine_tuned_df[["generated_size", "rouge1", "rouge2", "rougeL", "rougeLsum"]])
    print("Averages by generated output size:")
    print(averages)
    plot_by_group_sizes(averages, output_dir)

    # Compute semantic similarities and plot the results
    model_ST = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Computing semantic similarity scores...")
    results["semantic_similarity"] = compute_cosine_similarities(results, model_ST)
    filtered_df = results[results["model"] == first_model]
    plot_similarity_vs_rouge(filtered_df, output_dir)
    plot_semantic_similarity_distribution(filtered_df, output_dir)
    
    logger.info("Finished analysis.")


if __name__ == "__main__":
    main()
