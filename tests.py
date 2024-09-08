import pandas as pd

def validate_classification(output_file, original_file, labels_file):

    df_output = pd.read_csv(output_file, header=None, names=['id', 'isco_code'])
    df_original = pd.read_csv(original_file)
    df_labels = pd.read_csv(labels_file)


    df_output['isco_code'] = df_output['isco_code'].astype(str)
    df_labels['code'] = df_labels['code'].astype(str)

    # Check if all ids in the original file are in the output file
    original_ids = set(df_original['id'].astype(str))
    output_ids = set(df_output['id'].astype(str))

    missing_ids = original_ids - output_ids
    if missing_ids:
        raise Exception(f"The following ids are missing in the output file: {missing_ids}")

    # Check if all ids in the output file are unique
    if df_output['id'].duplicated().any():
        duplicated_ids = df_output[df_output['id'].duplicated()]['id'].tolist()
        raise Exception(f"There are duplicate ids in the output file: {duplicated_ids}")

    # Check if there is only one ISCO code for each id
    if df_output.groupby('id')['isco_code'].nunique().max() > 1:
        problematic_ids = df_output[df_output.groupby('id')['isco_code'].transform('nunique') > 1]['id'].tolist()
        raise Exception(f"There is more than one ISCO code for the following ids in the output file: {problematic_ids}")

    # Check for empty ISCO codes and ISCO codes with less than 4 characters
    invalid_isco_codes = df_output[df_output['isco_code'].isnull() | (df_output['isco_code'].str.len() < 4)]
    if not invalid_isco_codes.empty:
        invalid_ids = invalid_isco_codes['id'].tolist()
        raise Exception(f"There are empty ISCO codes or ISCO codes with less than 4 characters for the following ids: {invalid_ids}")

    # Check if all ISCO codes in the output file are present in the labels file
    valid_isco_codes = set(df_labels['code'])
    invalid_isco_codes = df_output[~df_output['isco_code'].isin(valid_isco_codes)]
    if not invalid_isco_codes.empty:
        invalid_ids = invalid_isco_codes['id'].tolist()
        raise Exception(f"The following ids have invalid ISCO codes not found in the labels file: {invalid_ids}")

    print("Validation successful.")

try:
    validate_classification('data/results.csv', 'data/wi_dataset.csv', 'data/wi_labels.csv')
    print("Validation successful")
except Exception as e:
    print(f"Validation failed: {e}")
