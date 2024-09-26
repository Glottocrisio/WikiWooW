import json
from datetime import datetime
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import cohen_kappa_score
import pandas as pd

def read_manifest(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['source']: item for item in data}

def write_manifest(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
def adjust_source_format(source):
    return source.replace("'\t '", "- ").replace("';  '", "- ").replace("' ;  '", "- ").strip("'").replace("';'", "-").replace("'", "")

def correct_json_file(input_path, output_path):
    # Reading the file content
    with open(input_path, 'r') as file:
        data = json.load(file)
    
    # Adjusting the source format for each json object
    for json_obj in data:
        source = json_obj.get("source", "")
        json_obj["source"] = adjust_source_format(source)
    
    # Saving the adjusted content to a new JSON file
    with open(output_path, 'w') as output_file:
        json.dump(data, output_file, indent=4)
        

def calculate_fleiss_kappa(data1, data2):
    n = len(data1)  # number of items
    N = 2  # assume 10 annotators per item, adjust as needed

    # Create matrix where each row represents an item and each column represents a category
    matrix = np.zeros((n, 2))
    
    for i, (source, item1) in enumerate(data1.items()):
        if source in data2:
            
            conf1 = item1['interestingness-detector-metadata']['confidence']
            if conf1 > 0.5:
                conf1 = 1
            else:
                conf1 = 0
            # if data2[source]['Serendipity-dataset-metadata']['class-name'] != "Interesting/Serendipitous":
            #     conf2 = 1 - data2[source]['Serendipity-dataset-metadata']['confidence']
            # else:
            conf2 = data2[source]['Serendipity-dataset-metadata']['confidence']
            if conf2 > 0.5:
                conf2 = 1
            else:
                conf2 = 0
            # Convert confidence to number of annotators
            n1 = round(conf1 * N, 2)
            n2 = round(conf2 * N, 2)
            
            # Average number of annotators who rated as interesting
            n_interesting = (n1 + n2) / 2
            
            matrix[i, 0] = n_interesting
            matrix[i, 1] = N - n_interesting

    # Calculate P_i (proportion of agreeing pairs for each item)
    P_i = np.sum(matrix * (matrix - 1), axis=1) / (N * (N - 1))
    
    # Calculate P_bar (mean of P_i's)
    P_bar = np.mean(P_i)
    
    # Calculate P_e (expected agreement by chance)
    P_e = np.sum(np.sum(matrix, axis=0) ** 2) / (N * N * n)
    
    # Calculate Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa

def interpret_kappa(kappa):
    if kappa < 0:
        return "Poor agreement"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"
    
def combine_manifests(file1_path, file2_path, output_path):
    # Read both manifest files
    data1 = read_manifest(file1_path)
    data2 = read_manifest(file2_path)
    #kappa = calculate_fleiss_kappa(data1, data2)
    # Find common entries and calculate average confidence
    common_data = []
    for source in set(data1.keys()) & set(data2.keys()):
        entry1 = data1[source]
        entry2 = data2[source]
        if float(entry1['interestingness-detector-metadata']['confidence']) >  float(entry2['Serendipity-dataset-metadata']['confidence']):
            low_confidence = entry2['Serendipity-dataset-metadata']['confidence']
        else:
            low_confidence = entry1['interestingness-detector-metadata']['confidence']
        avg_confidence = (entry1['interestingness-detector-metadata']['confidence'] + 
                          entry2['Serendipity-dataset-metadata']['confidence']) / 2
        
        # Create new entry with averaged confidence
        new_entry = {
            "source": source,
            "Serendipity-dataset": 0,
            "Serendipity-dataset-metadata": {
                "class-name": "Interesting/Serendipitous",
                "job-name": "labeling-job/serendipity-dataset",
                "confidence": low_confidence,
                "type": "groundtruth/text-classification",
                "human-annotated": "yes",
                "creation-date": datetime.utcnow().isoformat()
            }
        }
        common_data.append(new_entry)

    # Write the new manifest file
    write_manifest(output_path, common_data)
    
    print(f"Combined manifest file has been created: {output_path}")

def fleiss_kappa(file1, file2):
    # Read both manifest files
    data1 = read_manifest(file1)
    data2 = read_manifest(file2)
    kappa = calculate_fleiss_kappa(data1, data2)
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"
    table.add_row(["Fleiss' Kappa", f"{kappa:.4f}"])
    table.add_row(["Interpretation", interpret_kappa(kappa)])
    table.add_row(["Total entries in File 1", len(data1)])
    table.add_row(["Total entries in File 2", len(data2)])
    
    print(table)

def cohen_kappa(file1_path, file2_path, threshold=0.8):
    # Read both manifest files
    data1 = read_manifest(file1_path)
    data2 = read_manifest(file2_path)

    # Find common entries
    common_sources = set(data1.keys()) & set(data2.keys())

    # Prepare lists for Cohen's Kappa calculation
    rater1 = []
    rater2 = []

    for source in common_sources:
        conf1 = data1[source]['interestingness-detector-metadata']['confidence']
        conf2 = data2[source]['Serendipity-dataset-metadata']['confidence']
        
        # Convert confidence to binary classification
        rater1.append(1 if conf1 >= threshold else 0)
        rater2.append(1 if conf2 >= threshold else 0)

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(rater1, rater2)


    return kappa

def intersection(file1_path, file2_path, output1_path, output2_path):
    # Read both manifest files
    data1 = read_manifest(file1_path)
    data2 = read_manifest(file2_path)

    # Find entries where confidence differs
    differing_entries = []
    for source in set(data1.keys()) & set(data2.keys()):
        conf1 = data1[source]['interestingness-detector-metadata']['confidence']
        conf2 = data2[source]['Serendipity-dataset-metadata']['confidence']
        if conf1 != conf2:
            differing_entries.append(source)

    # Create output data for file 1
    output1_data = [data1[source] for source in differing_entries]

    # Create output data for file 2
    output2_data = [data2[source] for source in differing_entries]

    # Write output files
    write_manifest(output1_path, output1_data)
    write_manifest(output2_path, output2_data)

    print(f"Number of entries with differing confidence: {len(differing_entries)}")
    print(f"Output file 1 created: {output1_path}")
    print(f"Output file 2 created: {output2_path}")

def csv_cohen_kappa(file_path, column1, column2):
    """
    Calculate Cohen's Kappa for two columns in a CSV file.
    
    Args:
    file_path (str): Path to the CSV file
    column1 (str): Name of the first column
    column2 (str): Name of the second column
    
    Returns:
    float: Cohen's Kappa score
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if both columns exist in the DataFrame
        if column1 not in df.columns or column2 not in df.columns:
            raise ValueError(f"One or both columns ({column1}, {column2}) not found in the CSV file.")
        
        # Extract the two columns
        rater1 = df[column1]
        rater2 = df[column2]
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(rater1, rater2)
        
        return kappa
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



file_path = "updated_data.csv"
column1 = "Int_Hum_Eval"
column2 = "ground truth (threshold 0.8)"
#Cohen's Kappa: -0.1038
#Interpretation: Poor agreement   
kappa_score = csv_cohen_kappa(file_path, column1, column2)
    
if kappa_score is not None:
    print(f"Cohen's Kappa score: {kappa_score:.4f}")

# Input file paths
file1_path = 'ground_truth_interestingness_valid.json'
file2_path = 'ground_truth_serendipity_valid_c.json'

# Output file path
output_path = 'combined_manifest_fl.json'

#correct_json_file('ground_truth_serendipity_valid.json', 'ground_truth_serendipity_valid_c.json')

# Process the manifests
#combine_manifests(file1_path, file2_path, output_path)

#intersection('ground_truth_interestingness_valid.json', 'ground_truth_serendipity_valid_c.json', 'intersection_ground_truth_interestingness_valid.json', 'intersection_ground_truth_serendipity_valid_c.json')
fleiss_kappa('intersection_ground_truth_interestingness_valid.json', 'intersection_ground_truth_serendipity_valid_c.json')


kappa = cohen_kappa('intersection_ground_truth_interestingness_valid.json', 'intersection_ground_truth_serendipity_valid_c.json')


print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Interpretation: {interpret_kappa(kappa)}")