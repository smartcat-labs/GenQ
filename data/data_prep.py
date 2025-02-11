import logging
import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") 

def load_data(dataset_name: str, subset_name: str)-> pd.DataFrame :
    """
    Loads a dataset from Hugging Face and converts it into a Pandas DataFrame.

    Args:
        dataset_name (str): The Hugging Face dataset repository name.
        subset_name (str): The specific dataset subset to load.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    logging.info(f"Loading dataset: {dataset_name}, subset={subset_name}")
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
    logging.info("Checking data quality...")

    # Check for missing values
    nan_counts = df.isnull().sum()
    logging.info(f"Missing values:\n{nan_counts[nan_counts > 0]}")

    # Check for empty lists
    empty_list_counts = df.apply(lambda col: col.apply(lambda x: x == []).sum())
    logging.info(f"Empty lists per column:\n{empty_list_counts[empty_list_counts > 0]}")

    # Check for empty strings
    empty_string_counts = (df == "").sum()
    logging.info(f"Empty strings per column:\n{empty_string_counts[empty_string_counts > 0]}")

    # Check for empty dictionaries
    empty_dict_counts = df.apply(lambda col: col.apply(lambda x: isinstance(x, dict) and len(x) == 0).sum())
    logging.info(f"Empty dictionaries per column:\n{empty_dict_counts[empty_dict_counts > 0]}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by:
    - Removing rows where 'title' or 'short_query' is empty
    - Filling missing values in 'main_category' with 'Unknown'
    - Filling missing values in 'brand' with 'store' values
    - Dropping unused columns 'store', 'manufacturer' and 'price'
    """
    logging.info("Cleaning data...")

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

    logging.info(f"Brand & Store match: {brand_store_match:.2f}%")
    logging.info(f"Brand & Manufacturer match: {brand_manufacturer_match:.2f}%")
    logging.info(f"Store & Manufacturer match: {store_manufacturer_match:.2f}%")

def analyze_ordinal_columns(df: pd.DataFrame, process_columns: list = None , skip_columns: list = None) -> None:
    """
    Analyzes specified columns in the DataFrame for unique values and occurrences.

    Parameters:
        - df (pd.DataFrame): The DataFrame to analyze.
        - process_columns (list): List of column names to process. If None, all columns except skip_columns will be processed.
        - skip_columns (list): List of column names to skip.

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

        logging.info("--------------------")
        logging.info(f"Analyzing column: '{column}'")
        logging.info("--------------------")

        # Number of unique values
        num_unique = df[column].nunique()
        logging.info(f"  Number of unique values: {num_unique}")

        # Unique values
        unique_values = df[column].unique()
        logging.info(f"  Unique values: {unique_values}")

        # Count occurrences of each unique value
        value_counts = df[column].value_counts()
        for value, count in value_counts.items():
            logging.info(f"  Number of occurrences of '{value}': {count}")

        logging.info("\n")  # Newline for readability between columns

def merge_additional_data(df: pd.DataFrame, dataset_name: str, subset_name:str) -> pd.DataFrame:
    """
    Merges additional dataset with the main dataset based on 'parent_asin'.

    Args:
        df (pd.DataFrame): The main dataset.
        dataset_name (str): The dataset repository to merge from.
        subset_name (str): The specific subset to load.

    Returns:
        pd.DataFrame: The merged dataset sorted.
    """
    logging.info(f"Merging dataset: {dataset_name}, subset={subset_name}")
    temp_ds = load_dataset(dataset_name, name=subset_name, split="train")
    temp_df = pd.DataFrame(temp_ds)

    merged_df = pd.merge(df, temp_df[['parent_asin', 'images', 'text']], on='parent_asin', how='left')
    merged_df = merged_df.rename(columns={'text': 'embellished_description'})

    return merged_df[['parent_asin', 'main_category', 'title', 'description', 'features', 'embellished_description', 'brand', 'images', 'short_query', 'long_query']]

def split_dataset(df: pd.DataFrame) -> DatasetDict:
    """
    Splits the dataset into Train (80%), Validation (10%), and Test (10%) sets.

    Returns:
        DatasetDict: The split dataset.
    """
    total_rows = len(df)
    train_len = int(total_rows * 0.8)
    validation_len = int(total_rows * 0.1)
    test_len = total_rows - (train_len + validation_len)

    logging.info(f"Dataset split sizes: Train={train_len}, Validation={validation_len}, Test={test_len}")

    # Shuffle dataset before splitting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df[:train_len]
    validation_df = df[train_len:train_len + validation_len]
    test_df = df[train_len + validation_len:]

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)
    test_dataset = Dataset.from_pandas(test_df)

    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

def push_to_huggingface(dataset: DatasetDict, repo_name: str) -> None:
    """
    Pushes the dataset to Hugging Face Hub.

    Args:
        dataset (DatasetDict): The dataset to push.
        repo_name (str): The repository name on Hugging Face.
    """
    logging.info(f"Pushing dataset to Hugging Face: {repo_name}")
    dataset.push_to_hub(repo_name)
    logging.info("Dataset successfully uploaded.")

def main():
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
    final_dataset = split_dataset(df)

    # Push dataset to Hugging Face
    push_to_huggingface(final_dataset, "smartcat/Amazon-2023-GenQ")

if __name__ == "__main__":
    main()