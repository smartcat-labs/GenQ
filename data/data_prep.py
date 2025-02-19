from loguru import logger
from typing import List
import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
from sklearn.model_selection import train_test_split


def load_data(dataset_name: str, subset_name: str)-> pd.DataFrame :
    """
    Loads a dataset from Hugging Face and converts it into a Pandas DataFrame.

    Args:
        dataset_name (str): The Hugging Face dataset repository name.
        subset_name (str): The specific dataset subset to load.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    logger.info(f"Loading dataset: {dataset_name}, subset={subset_name}")
    dataset = load_dataset(dataset_name, name=subset_name, split="train")
    df = pd.DataFrame(dataset)
    
    # Move 'parent_asin' to first column for better organization
    new_column_order = ["parent_asin"] + [col for col in df.columns if col != "parent_asin"]
    df = df[new_column_order]
    
    return df

def check_data_quality(df: pd.DataFrame) -> None:
    """
    Checks for missing values, empty strings, empty lists, and empty dictionaries in the dataset.
    """
    logger.info("Checking data quality...")

    # Check for missing values
    nan_counts = df.isnull().sum()
    logger.info(f"Missing values:\n{nan_counts[nan_counts > 0]}")

    # Check for empty lists
    empty_list_counts = df.apply(lambda col: col.apply(lambda x: x == []).sum())
    logger.info(f"Empty lists per column:\n{empty_list_counts[empty_list_counts > 0]}")

    # Check for empty strings
    empty_string_counts = (df == "").sum()
    logger.info(f"Empty strings per column:\n{empty_string_counts[empty_string_counts > 0]}")

    # Check for empty dictionaries
    empty_dict_counts = df.apply(lambda col: col.apply(lambda x: isinstance(x, dict) and len(x) == 0).sum())
    logger.info(f"Empty dictionaries per column:\n{empty_dict_counts[empty_dict_counts > 0]}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Removing rows where 'title' or 'short_query' is empty
    - Filling missing values in 'main_category' with 'Unknown'
    - Filling missing values in 'brand' with 'store' values
    - Dropping unused columns 'store', 'manufacturer' and 'price'
    """
    logger.info("Cleaning data...")

    # Remove rows where 'title' is empty
    df = df[df["title"] != ""]

    # Remove rows where 'short_query' is empty
    df = df[df["short_query"] != ""]

    # Replace empty 'main_category' values with 'Unknown'
    df.loc[df["main_category"] == "", "main_category"] = "Unknown"

    df["brand"] = df["brand"].fillna(df["store"])

    df.drop(columns=["store", "manufacturer", "price"], inplace=True)

    return df

def compute_column_similarity(df: pd.DataFrame) -> None:
    """
    Computes the percentage of matching values between brand, store, and manufacturer.
    """
    df_lower = df.astype(str).apply(lambda x: x.str.lower())

    total_rows = len(df)
    brand_store_match = (df_lower["brand"] == df_lower["store"]).sum() / total_rows * 100
    brand_manufacturer_match = (df_lower["brand"] == df_lower["manufacturer"]).sum() / total_rows * 100
    store_manufacturer_match = (df_lower["store"] == df_lower["manufacturer"]).sum() / total_rows * 100

    logger.info(f"Brand & Store match: {brand_store_match:.2f}%")
    logger.info(f"Brand & Manufacturer match: {brand_manufacturer_match:.2f}%")
    logger.info(f"Store & Manufacturer match: {store_manufacturer_match:.2f}%")

def analyze_ordinal_columns(df: pd.DataFrame, process_columns: List[str] = None , skip_columns: List[str] = None) -> None:
    """
    Analyzes specified columns in the DataFrame for unique values and occurrences.

    Parameters:
        - df (pd.DataFrame): The DataFrame to analyze.
        - process_columns (List[str]): List of column names to process. If None, all columns except skip_columns will be processed.
        - skip_columns (List[str]): List of column names to skip.

    Logs:
        - Number of unique values in each processed column.
        - Unique values in each processed column.
        - Number of occurrences for each unique value in each processed column.
    """
    if skip_columns is None:
        skip_columns = []
    if process_columns is None:
        process_columns = df.columns.tolist()

    # Process the specified columns
    for column in process_columns:
        if column in skip_columns:
            continue  # Skip the column if it is in the skip list

        logger.info("--------------------")
        logger.info(f"Analyzing column: '{column}'")
        logger.info("--------------------")

        # Number of unique values
        num_unique = df[column].nunique()
        logger.info(f"Number of unique values: {num_unique}")

        # Unique values
        unique_values = df[column].unique()
        logger.info(f"Unique values: {unique_values}")

        # Count occurrences of each unique value
        value_counts = df[column].value_counts()
        for value, count in value_counts.items():
            logger.info(f"Number of occurrences of '{value}': {count}")

def merge_additional_data(df: pd.DataFrame, dataset_name: str, subset_name: str) -> pd.DataFrame:
    """
    Merges additional dataset with the main dataset based on 'parent_asin'.

    Args:
        df (pd.DataFrame): The main dataset.
        dataset_name (str): The dataset repository to merge from.
        subset_name (str): The specific subset to load.

    Returns:
        pd.DataFrame: The merged dataset sorted.
    """
    logger.info(f"Merging dataset: {dataset_name}, subset={subset_name}")
    temp_ds = load_dataset(dataset_name, name=subset_name, split="train")
    temp_df = pd.DataFrame(temp_ds)

    merged_df = pd.merge(df, temp_df[['parent_asin', 'images', 'text']], on='parent_asin', how='left')
    merged_df = merged_df.rename(columns={'text': 'embellished_description'})

    return merged_df[['parent_asin', 'main_category', 'title', 'description', 'features', 'embellished_description', 'brand', 'images', 'short_query', 'long_query']]

def split_dataset(df: pd.DataFrame, train_ratio: float = None, val_ratio: float = None, test_ratio: float = None, seed: int = 42) -> DatasetDict:
    """
    Splits the dataset into Train, Validation and Test sets.

    Agrs:
        df (pd.DataFrame): The dataset to be split.  
        train_ratio (float): The percentage for the train split
        val_ratio: (float): The percentage for the validation split
        test_ratio: (float): The percentage for the test_ratio split

    Returns:
        DatasetDict: The split dataset.

    Note:
        If the provided ratios do not sum to 1, default values will be used.
    """

    ratios = {"train": train_ratio, "validation": val_ratio, "test": test_ratio}
    unset_keys = [key for key, value in ratios.items() if value is None]

    if len(unset_keys) == 1:
        # Compute the missing value
        set_keys = [key for key in ratios if key not in unset_keys]
        ratios[unset_keys[0]] = 1.0 - sum(ratios[k] for k in set_keys)

        if not (0 <= ratios[unset_keys[0]] <= 1):
            raise ValueError(f"Invalid split values: {ratios}. The computed {unset_keys[0]} ratio is out of bounds.")

        logger.info(f"Automatically setting {unset_keys[0]} ratio to {ratios[unset_keys[0]]:.2f}.")

    elif len(unset_keys) > 1 or abs(sum(ratios.values()) - 1.0) > 1e-6:
        logger.warning("Ratios are invalid or incomplete. Defaulting to 80% train, 10% validation, and 10% test.")
        ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}

    # Extract final ratios
    train_ratio, val_ratio, test_ratio = ratios["train"], ratios["validation"], ratios["test"]


    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=seed)
    validation_df, test_df = train_test_split(temp_df, train_size=val_ratio / (val_ratio + test_ratio), random_state=seed)

    logger.info(f"Dataset split sizes: Train={len(train_df)}, Validation={len(validation_df)}, Test={len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)
    test_dataset = Dataset.from_pandas(test_df)

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

def main():
    # Configure logging
    logger.add("data/preprocess.log", level="INFO")

    # Load dataset
    df = load_data("smartcat/Amazon_Sample_Metadata_2023", "product2query_V1")

    # Check data quality
    check_data_quality(df)

    # Compute column similarity (brand, store, manufacturer)
    compute_column_similarity(df)

    # Clean the dataset
    df = clean_data(df)

    #Analyze the main_category column
    analyze_ordinal_columns(df, ["main_category"])

    # Merge additional dataset with descriptions
    df = merge_additional_data(df, "smartcat/Amazon_Sample_Metadata_2023", "combined_description_formatted")

    # Split dataset into train, validation, and test
    final_dataset = split_dataset(df, train_ratio=0.8, test_ratio=0.1)

    # Push dataset to Hugging Face
    logger.info("Pushing dataset to Hugging Face..")
    final_dataset.push_to_hub("smartcat/Amazon-2023-GenQ")
    logger.info('Dataset successfully uploaded to "smartcat/Amazon-2023-GenQ".')

if __name__ == "__main__":
    main()