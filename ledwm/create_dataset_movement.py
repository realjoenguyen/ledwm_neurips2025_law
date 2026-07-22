# %%
import json
import pandas as pd


def json_to_dataset(json_file_path, output_csv_path):
    # Load the JSON data
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # Prepare lists for dataset
    texts = []
    categories = []

    # Iterate through the JSON structure to extract text samples and their corresponding categories
    for main_category, subcategories in data.items():
        for subcategory, category_data in subcategories.items():
            for category, samples in category_data.items():
                for sample in samples:
                    texts.append(sample)
                    categories.append(category.capitalize())

    # Create a DataFrame from the collected data
    df = pd.DataFrame({"Text": texts, "Category": categories})

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Dataset saved to {output_csv_path}")


# Example usage
json_file_path = "messenger-emma/messenger/envs/texts/text_val.json"
output_csv_path = "val_movement.csv"  # Replace with your desired output CSV file path
json_to_dataset(json_file_path, output_csv_path)
