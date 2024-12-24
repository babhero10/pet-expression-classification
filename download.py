"""
Downloads and prepares the dataset for use.
"""
import os
import shutil
import kaggle

def download_and_prepare_dataset(dataset_name="anshtanwar/pets-facial-expression-dataset", download_dir="data"):
    """
    Downloads the dataset from Kaggle, removes unnecessary folders, renames Master Folder to pet_expression_classification.
    """
    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading dataset '{dataset_name}' to '{download_dir}'...")
    kaggle.api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
    
    # Path to the downloaded data
    dataset_path = os.path.join(download_dir, "Master Folder")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Master Folder not found in the downloaded dataset.")

    # Remove other folders and keep only the Master Folder
    for item in os.listdir(download_dir):
        item_path = os.path.join(download_dir, item)
        if item_path != dataset_path:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    # Rename Master Folder to pet_expression_classification
    new_path = os.path.join(download_dir, "pet_expression_classification")
    os.rename(dataset_path, new_path)

    print(f"Dataset prepared at: {new_path}")
    return new_path

if __name__ == "__main__":
    download_and_prepare_dataset()
