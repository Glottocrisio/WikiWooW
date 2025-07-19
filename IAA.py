# Different functions to compute the inter-annotator agreement metrics from the Google Form containing
# the annotator's answers on questions related to perceived relatedness, subjective knowledge and Serendipity 
# of the entity pairs.


import json
from datetime import datetime
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from krippendorff import alpha
import csv
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import statsmodels.stats.inter_rater as ir
from itertools import combinations

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



# file_path = "updated_data.csv"
# column1 = "Int_Hum_Eval"
# column2 = "ground truth (threshold 0.8)"
# #Cohen's Kappa: -0.1038
# #Interpretation: Poor agreement   
# kappa_score = csv_cohen_kappa(file_path, column1, column2)
    
# if kappa_score is not None:
#     print(f"Cohen's Kappa score: {kappa_score:.4f}")

# # Input file paths
# file1_path = 'ground_truth_interestingness_valid.json'
# file2_path = 'ground_truth_serendipity_valid_c.json'

# # Output file path
# output_path = 'combined_manifest_fl.json'

# #correct_json_file('ground_truth_serendipity_valid.json', 'ground_truth_serendipity_valid_c.json')

# # Process the manifests
# #combine_manifests(file1_path, file2_path, output_path)

# #intersection('ground_truth_interestingness_valid.json', 'ground_truth_serendipity_valid_c.json', 'intersection_ground_truth_interestingness_valid.json', 'intersection_ground_truth_serendipity_valid_c.json')
# fleiss_kappa('intersection_ground_truth_interestingness_valid.json', 'intersection_ground_truth_serendipity_valid_c.json')


# kappa = cohen_kappa('intersection_ground_truth_interestingness_valid.json', 'intersection_ground_truth_serendipity_valid_c.json')


# print(f"Cohen's Kappa: {kappa:.4f}")
# print(f"Interpretation: {interpret_kappa(kappa)}")

def is_categorical_column(series):
    """
    Check if a column is categorical (e.g., Yes/No/DOES NOT APPLY)
    """
    # Convert to strings and get unique values (excluding NaN)
    values = series.dropna().astype(str).str.lower().unique()
    
    # Check if values match common categorical patterns
    categorical_patterns = [
        # Yes/No pattern
        (set(['yes', 'no']) - set(values) == set() or 
         set(['yes', 'no', 'does not apply']) - set(values) == set()),
        # True/False pattern
        (set(['true', 'false']) - set(values) == set()),
        # 0/1 as categorical
        (set(['0', '1']) == set(values)),
        # Very small number of unique values (likely categorical)
        len(values) <= 5
    ]
    
    return any(categorical_patterns)

def calculate_krippendorff_alpha(csv_file_path):
    """
    Calculate Krippendorff's alpha for interannotator agreement from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        float: Krippendorff's alpha value
    """
    # Read the CSV file with semicolon delimiter
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='ISO-8859-1')
    
    # Display basic information about the dataset
    print(f"Total number of entries: {len(df)}")
    print(f"Number of annotators: {df['Annotator'].nunique()}")
    print(f"Number of entity pairs: {df['Entity Couple'].nunique()}")
    
    # Check which column to use for annotation
    if 'Annotation' in df.columns:
        annotation_column = 'Annotation'
    elif 'S_corrected_Annotation' in df.columns:
        annotation_column = 'S_corrected_Annotation'
    else:
        print("No annotation column found. Please specify the correct column.")
        return None
    
    print(f"Using '{annotation_column}' column for calculations")
    
    # Convert comma to dot for decimal numbers if needed
    df[annotation_column] = df[annotation_column].astype(str).str.replace(',', '.').astype(float)
    
    # Reshape data for Krippendorff's alpha calculation
    # Create a matrix where rows are items and columns are annotators
    pivot_df = df.pivot(index='Entity Couple', columns='Annotator', values=annotation_column)
    print(f"Pivot table shape: {pivot_df.shape}")
    # Convert to a reliability data matrix for krippendorff alpha calculation
    reliability_data = np.array(pivot_df.values.tolist())
    print(f"Reliability data shape: {reliability_data.shape}")
    # Calculate Krippendorff's alpha
    # Note: krippendorff.alpha expects a reliability data matrix where columns are coders and rows are units
    k_alpha = alpha(reliability_data=reliability_data.T, level_of_measurement='interval')
    
    print(f"\nKrippendorff's Alpha: {k_alpha:.4f}")
    
    # Interpretation guide
    if k_alpha < 0:
        interpretation = "Poor agreement (less than chance)"
    elif k_alpha < 0.2:
        interpretation = "Slight agreement"
    elif k_alpha < 0.4:
        interpretation = "Fair agreement"
    elif k_alpha < 0.6:
        interpretation = "Moderate agreement"
    elif k_alpha < 0.8:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"Interpretation: {interpretation}")
    
    return k_alpha
   
# Calculate Krippendorff's alpha
k_alpha = calculate_krippendorff_alpha('Form Responses Export 2025-05-04T14_31_37.049Z.csv')
    
def calculate_agreement_metrics(csv_file_path, annotation_column=None, round_values=True, round_precision=1):
    """
    Calculate multiple interannotator agreement metrics from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        annotation_column (str, optional): Column name for annotations. If None, it will be auto-detected.
        round_values (bool): Whether to round annotation values for categorical metrics
        round_precision (int): Precision for rounding (default: 1 decimal place)
        
    Returns:
        dict: Dictionary of agreement metrics
    """
    # Read the CSV file with semicolon delimiter
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='ISO-8859-1')
    
    # Display basic information about the dataset
    print(f"Total number of entries: {len(df)}")
    print(f"Number of annotators: {df['Annotator'].nunique()}")
    print(f"Number of entity pairs: {df['Entity Couple'].nunique()}")
    print(f"Annotators: {sorted(df['Annotator'].unique())}")
    
    # Check which column to use for annotation
    if annotation_column is not None:
        if annotation_column not in df.columns:
            print(f"Column '{annotation_column}' not found. Available columns are: {df.columns.tolist()}")
            return None
    else:
        if 'Annotation' in df.columns:
            annotation_column = 'Annotation'
        elif 'S_corrected_Annotation' in df.columns:
            annotation_column = 'S_corrected_Annotation'
        else:
            print("No annotation column found. Please specify the correct column.")
            print(f"Available columns are: {df.columns.tolist()}")
            return None
    
    print(f"Using '{annotation_column}' column for calculations")
    
    # Convert comma to dot for decimal numbers if needed
    df[annotation_column] = df[annotation_column].astype(str).str.replace(',', '.').astype(float)
    
    # Reshape data for agreement calculations
    # Create a matrix where rows are items and columns are annotators
    pivot_df = df.pivot(index='Entity Couple', columns='Annotator', values=annotation_column)
    print("\nData matrix (rows: items, columns: annotators):")
    print(pivot_df.head())
    
    # Create a copy of the data for categorical metrics if rounding is requested
    if round_values:
        # Round to the specified precision
        rounded_pivot = pivot_df.round(round_precision)
        print(f"\nRounded data (precision: {round_precision} decimal places):")
        print(rounded_pivot.head())
    else:
        rounded_pivot = pivot_df.copy()
    
    # Dictionary to store all metrics
    metrics = {}
    
    # 1. Intraclass Correlation Coefficient (ICC)
    try:
        icc = ir.icc(pivot_df.values, model='twoway', type='agreement')
        metrics['ICC'] = {
            'value': icc[0],
            'description': 'Two-way random effects, agreement (ICC2)'
        }
        print(f"\n1. ICC (Two-way random effects, agreement): {icc[0]:.4f}")
    except Exception as e:
        print(f"\n1. ICC calculation error: {str(e)}")
        metrics['ICC'] = {'value': None, 'error': str(e)}
    
    # 2. Fleiss' Kappa (for multiple raters with categorical data)
    try:
        # Convert to categorical data for Fleiss' Kappa
        categorical_data = rounded_pivot.copy()
        
        # Get unique categories
        all_values = categorical_data.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        categories = sorted(set(all_values))
        
        # Prepare data for Fleiss' Kappa
        n_categories = len(categories)
        n_items = len(categorical_data)
        n_raters = len(categorical_data.columns)
        
        # Create a matrix where each row represents an item and each column represents a category
        # The value in each cell is the number of raters who assigned that category to that item
        m = np.zeros((n_items, n_categories))
        
        for item_idx, item in enumerate(categorical_data.index):
            ratings = categorical_data.loc[item].values
            ratings = ratings[~np.isnan(ratings)]
            for category_idx, category in enumerate(categories):
                m[item_idx, category_idx] = sum(ratings == category)
        
        # Calculate Fleiss' Kappa
        p = m / m.sum(axis=1)[:, np.newaxis]
        P = (p * p).sum(axis=1)
        P_mean = P.mean()
        p_mean = p.mean(axis=0)
        PE = (p_mean * p_mean).sum()
        kappa = (P_mean - PE) / (1 - PE)
        
        metrics['Fleiss_Kappa'] = {
            'value': kappa,
            'description': f'Agreement between {n_raters} raters with {n_categories} categories'
        }
        print(f"\n2. Fleiss' Kappa: {kappa:.4f}")
        print(f"   Categories: {categories}")
    except Exception as e:
        print(f"\n2. Fleiss' Kappa calculation error: {str(e)}")
        metrics['Fleiss_Kappa'] = {'value': None, 'error': str(e)}
    
    # 3. Pairwise Cohen's Kappa (for each pair of annotators with categorical data)
    annotators = pivot_df.columns
    pairwise_kappa = {}
    
    try:
        print("\n3. Pairwise Cohen's Kappa:")
        for a1, a2 in combinations(annotators, 2):
            # Extract ratings for both annotators
            ratings1 = rounded_pivot[a1].values
            ratings2 = rounded_pivot[a2].values
            
            # Keep only rows where both annotators provided ratings
            mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
            if sum(mask) < 2:
                print(f"   {a1} vs {a2}: Not enough data")
                continue
                
            r1 = ratings1[mask]
            r2 = ratings2[mask]
            
            # Calculate Cohen's Kappa
            try:
                kappa = cohen_kappa_score(r1, r2)
                pairwise_kappa[f"{a1}_vs_{a2}"] = kappa
                print(f"   {a1} vs {a2}: {kappa:.4f}")
            except Exception as e:
                print(f"   {a1} vs {a2}: Error - {str(e)}")
        
        # Calculate average pairwise Kappa
        if pairwise_kappa:
            avg_kappa = sum(pairwise_kappa.values()) / len(pairwise_kappa)
            metrics['Average_Cohens_Kappa'] = {
                'value': avg_kappa,
                'description': 'Average of pairwise Cohen\'s Kappa values'
            }
            metrics['Pairwise_Cohens_Kappa'] = pairwise_kappa
            print(f"   Average pairwise Kappa: {avg_kappa:.4f}")
        else:
            metrics['Average_Cohens_Kappa'] = {'value': None, 'error': 'No valid pairs'}
    except Exception as e:
        print(f"\n3. Pairwise Cohen's Kappa calculation error: {str(e)}")
        metrics['Average_Cohens_Kappa'] = {'value': None, 'error': str(e)}
    
    # 4. Cronbach's Alpha (reliability measure)
    try:
        # Calculate item variance and total variance
        item_variances = np.nanvar(pivot_df.values, axis=1, ddof=1)
        total_variance = np.nanvar(pivot_df.values.flatten(), ddof=1)
        n_items = len(pivot_df)
        n_raters = len(pivot_df.columns)
        
        # Calculate Cronbach's Alpha
        cronbach = (n_raters / (n_raters - 1)) * (1 - (np.nansum(item_variances) / (n_items * total_variance)))
        
        metrics['Cronbachs_Alpha'] = {
            'value': cronbach,
            'description': 'Internal consistency reliability measure'
        }
        print(f"\n4. Cronbach's Alpha: {cronbach:.4f}")
    except Exception as e:
        print(f"\n4. Cronbach's Alpha calculation error: {str(e)}")
        metrics['Cronbachs_Alpha'] = {'value': None, 'error': str(e)}
    
    # 5. Mean Squared Error between annotators
    try:
        print("\n5. Mean Squared Error between annotators:")
        pairwise_mse = {}
        
        for a1, a2 in combinations(annotators, 2):
            # Extract ratings for both annotators
            ratings1 = pivot_df[a1].values
            ratings2 = pivot_df[a2].values
            
            # Keep only rows where both annotators provided ratings
            mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
            if sum(mask) < 2:
                print(f"   {a1} vs {a2}: Not enough data")
                continue
                
            r1 = ratings1[mask]
            r2 = ratings2[mask]
            
            # Calculate MSE
            mse = mean_squared_error(r1, r2)
            pairwise_mse[f"{a1}_vs_{a2}"] = mse
            print(f"   {a1} vs {a2}: {mse:.4f}")
        
        # Calculate average pairwise MSE
        if pairwise_mse:
            avg_mse = sum(pairwise_mse.values()) / len(pairwise_mse)
            metrics['Average_MSE'] = {
                'value': avg_mse,
                'description': 'Average Mean Squared Error between annotators (lower is better)'
            }
            metrics['Pairwise_MSE'] = pairwise_mse
            print(f"   Average MSE: {avg_mse:.4f}")
        else:
            metrics['Average_MSE'] = {'value': None, 'error': 'No valid pairs'}
    except Exception as e:
        print(f"\n5. MSE calculation error: {str(e)}")
        metrics['Average_MSE'] = {'value': None, 'error': str(e)}
    
    # 6. Percent Agreement (exact match)
    try:
        print("\n6. Percent Agreement (exact match):")
        pairwise_agreement = {}
        
        for a1, a2 in combinations(annotators, 2):
            # Extract ratings for both annotators
            ratings1 = rounded_pivot[a1].values
            ratings2 = rounded_pivot[a2].values
            
            # Keep only rows where both annotators provided ratings
            mask = ~np.isnan(ratings1) & ~np.isnan(ratings2)
            if sum(mask) < 2:
                print(f"   {a1} vs {a2}: Not enough data")
                continue
                
            r1 = ratings1[mask]
            r2 = ratings2[mask]
            
            # Calculate exact agreement
            exact_matches = sum(r1 == r2)
            total = len(r1)
            percent_agreement = exact_matches / total if total > 0 else 0
            
            pairwise_agreement[f"{a1}_vs_{a2}"] = percent_agreement
            print(f"   {a1} vs {a2}: {percent_agreement:.4f} ({exact_matches}/{total} matches)")
        
        # Calculate average pairwise agreement
        if pairwise_agreement:
            avg_agreement = sum(pairwise_agreement.values()) / len(pairwise_agreement)
            metrics['Average_Percent_Agreement'] = {
                'value': avg_agreement,
                'description': 'Average percentage of exact matches between annotators'
            }
            metrics['Pairwise_Percent_Agreement'] = pairwise_agreement
            print(f"   Average Percent Agreement: {avg_agreement:.4f}")
        else:
            metrics['Average_Percent_Agreement'] = {'value': None, 'error': 'No valid pairs'}
    except Exception as e:
        print(f"\n6. Percent Agreement calculation error: {str(e)}")
        metrics['Average_Percent_Agreement'] = {'value': None, 'error': str(e)}
    
    # Summarize results
    print("\nSummary of Interannotator Agreement Metrics:")
    for metric_name, metric_info in metrics.items():
        if 'value' in metric_info and metric_info['value'] is not None:
            print(f"- {metric_name}: {metric_info['value']:.4f}")
    
    return metrics

calculate_agreement_metrics('Form Responses Export 2025-05-04T14_31_37.049Z.csv', annotation_column='S_corrected_Annotation', round_values=True, round_precision=1)

def extract_high_agreement_pairs(csv_file_path, output_file=None, annotation_column=None, 
                                agreement_threshold=0.8, std_dev_threshold=0.1, 
                                max_diff_threshold=0.3):
    """
    Extract entity pairs where annotators show high agreement and save to a file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        output_file (str): Path to save the output file (default: 'high_agreement_pairs.csv')
        annotation_column (str): Column name for annotations (if None, auto-detect)
        agreement_threshold (float): Percentage of annotator pairs that must agree
        std_dev_threshold (float): Maximum standard deviation allowed for annotations
        max_diff_threshold (float): Maximum difference allowed between any two annotations
        
    Returns:
        DataFrame: The extracted entity pairs with high agreement
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = 'high_agreement_pairs.csv'
    
    # Read the CSV file with semicolon delimiter
    df = pd.read_csv(csv_file_path, delimiter=';')
    
    # Display basic information about the dataset
    print(f"Total number of entries: {len(df)}")
    print(f"Number of annotators: {df['Annotator'].nunique()}")
    print(f"Number of unique entity pairs: {df['Entity Couple'].nunique()}")
    
    # Check which column to use for annotation
    if annotation_column is not None:
        if annotation_column not in df.columns:
            print(f"Column '{annotation_column}' not found. Available columns are: {df.columns.tolist()}")
            return None
    else:
        if 'Annotation' in df.columns:
            annotation_column = 'Annotation'
        elif 'S_corrected_Annotation' in df.columns:
            annotation_column = 'S_corrected_Annotation'
        else:
            print("No annotation column found. Please specify the correct column.")
            print(f"Available columns are: {df.columns.tolist()}")
            return None
    
    print(f"Using '{annotation_column}' column for calculations")
    
    # Convert comma to dot for decimal numbers if needed
    df[annotation_column] = df[annotation_column].astype(str).str.replace(',', '.').astype(float)
    
    # Reshape data for agreement calculations
    # Create a matrix where rows are items and columns are annotators
    pivot_df = df.pivot(index='Entity Couple', columns='Annotator', values=annotation_column)
    
    # Create a DataFrame to store agreement metrics for each entity pair
    agreement_df = pd.DataFrame(index=pivot_df.index)
    
    # Calculate standard deviation for each entity pair
    agreement_df['std_dev'] = pivot_df.std(axis=1)
    
    # Calculate max difference for each entity pair
    agreement_df['max_diff'] = pivot_df.max(axis=1) - pivot_df.min(axis=1)
    
    # Calculate pairwise agreement for each entity pair
    annotators = pivot_df.columns
    n_annotators = len(annotators)
    n_pairs = n_annotators * (n_annotators - 1) // 2  # Total number of annotator pairs
    
    # Define what constitutes agreement between two annotators
    def calculate_pair_agreement(row):
        agreements = 0
        total_pairs = 0
        
        for a1, a2 in combinations(annotators, 2):
            # Skip if either value is NaN
            if pd.isna(row[a1]) or pd.isna(row[a2]):
                continue
                
            # Count as agreement if difference is small
            if abs(row[a1] - row[a2]) <= max_diff_threshold:
                agreements += 1
            total_pairs += 1
        
        return agreements / total_pairs if total_pairs > 0 else np.nan
    
    agreement_df['pairwise_agreement'] = pivot_df.apply(calculate_pair_agreement, axis=1)
    
    # Extract entity pairs that meet the agreement criteria
    high_agreement_pairs = agreement_df[
        (agreement_df['std_dev'] <= std_dev_threshold) & 
        (agreement_df['pairwise_agreement'] >= agreement_threshold)
    ]
    
    # Sort by agreement level (highest first)
    high_agreement_pairs = high_agreement_pairs.sort_values(
        by=['pairwise_agreement', 'std_dev'], 
        ascending=[False, True]
    )
    
    # Get the original data for these high agreement pairs
    high_agreement_data = df[df['Entity Couple'].isin(high_agreement_pairs.index)]
    
    # Group by entity pair and calculate mean annotation
    entity_pair_stats = high_agreement_data.groupby('Entity Couple').agg({
        annotation_column: ['mean', 'std', 'min', 'max', 'count']
    })
    
    # Flatten the column hierarchy
    entity_pair_stats.columns = [f"{annotation_column}_{col}" for col in ['mean', 'std', 'min', 'max', 'count']]
    
    # Merge with the high agreement data
    high_agreement_pairs = high_agreement_pairs.merge(
        entity_pair_stats, left_index=True, right_index=True
    )
    
    # Reset index to have Entity Couple as a column
    high_agreement_pairs = high_agreement_pairs.reset_index()
    
    # Save to CSV
    high_agreement_pairs.to_csv(output_file, index=False, sep=';')
    
    # Print summary
    print(f"\nFound {len(high_agreement_pairs)} entity pairs with high agreement")
    print(f"Results saved to: {output_file}")
    
    # Also extract the actual annotations for these pairs for reference
    high_agreement_annotations = df[df['Entity Couple'].isin(high_agreement_pairs['Entity Couple'])]
    
    # Save the original annotations for high agreement pairs
    annotations_output = output_file.replace('.csv', '_annotations.csv')
    high_agreement_annotations.to_csv(annotations_output, index=False, sep=';')
    print(f"Original annotations for these pairs saved to: {annotations_output}")
    
    return high_agreement_pairs

        
# Extract high agreement entity pairs
# extract_high_agreement_pairs('Form Responses Export 2025-05-04T14_31_37.049Z.csv')

calculate_agreement_metrics('Form Responses Export 2025-05-04T14_31_37.049Z.csv')