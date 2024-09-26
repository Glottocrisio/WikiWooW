import pandas as pd
import plotly.graph_objects as go
import numpy as np

def load_data(file_path, delimiter):
    try:
        df = pd.read_csv(file_path, sep=delimiter)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def to_numeric_safe(series):
    return pd.to_numeric(series, errors='coerce')

def add_palma_interestingness_bool(df):
    if 'PalmaInterestingnessEnt1Ent2' in df.columns:
        datarr = df['PalmaInterestingnessEnt1Ent2'].dropna().values
        if len(datarr) > 0:
            median = np.median(datarr)
            df['PalmaInterestingnessBool'] = df['PalmaInterestingnessEnt1Ent2'].apply(lambda x: 1 if x > median else 0)
            print("Added 'PalmaInterestingnessBool' column")
        else:
            print("Warning: 'PalmaInterestingnessEnt1Ent2' column is empty or all NaN")
    else:
        print("Warning: 'PalmaInterestingnessEnt1Ent2' column not found in the dataset")
    return df

def create_parallel_coordinates_plot(df, output_filename):
    # Print column names for debugging
    print("Columns in the dataset:")
    print(df.columns)

    # Function to safely get column name
    def get_column(possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # Map of expected column names to possible variations
    column_map = {
        'ClickstreamEnt1Ent2': ['ClickstreamEnt1Ent2', 'Clickstream'],
        'PopularityEnt1': ['PopularityEnt1', 'Popularity1'],
        'PopularityEnt2': ['PopularityEnt2', 'Popularity2'],
        'PopularityDiff': ['PopularityDiff', 'PopDiff'],
        'PopularitySum': ['PopularitySum', 'PopSum'],
        'CosineSimilarityEnt1Ent2': ['CosineSimilarityEnt1Ent2', 'CosineSimilarity'],
        'DBpediaSimilarityEnt1Ent2': ['DBpediaSimilarityEnt1Ent2', 'DBpediaSimilarity'],
        'DBpediaRelatednessEnt1Ent2': ['DBpediaRelatednessEnt1Ent2', 'DBpediaRelatedness'],
        'PalmaInterestingnessEnt1Ent2': ['PalmaInterestingnessEnt1Ent2', 'PalmaInterestingness'],
        'Int_Hum_Eval': ['Int_Hum_Eval', 'HumanEvaluation'],
        'PalmaInterestingnessBool': ['PalmaInterestingnessBool']
    }

    # Create dimensions list for parallel coordinates
    dimensions = []
    for key, possible_names in column_map.items():
        col_name = get_column(possible_names)
        if col_name:
            numeric_series = to_numeric_safe(df[col_name])
            if not numeric_series.isna().all():  # Check if we have any non-NaN values
                min_val = numeric_series.min()
                max_val = numeric_series.max()
                if pd.notna(min_val) and pd.notna(max_val):
                    dimensions.append(
                        dict(range=[min_val, max_val],
                             label=key,
                             values=numeric_series)
                    )
            else:
                print(f"Warning: Column '{col_name}' could not be converted to numeric.")

    # Check if we have any dimensions to plot
    if not dimensions:
        print("No valid numeric columns found. Please check your data structure.")
        return

    # Create the parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=dimensions[0]['values'],
                      colorscale='Viridis',
                      showscale=True),
            dimensions=dimensions
        )
    )

    # Update layout
    fig.update_layout(
        title='Parallel Coordinates Plot of Entity Relationships',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Show the plot
    fig.show()

    # Save the figure as an HTML file
    fig.write_html(f"{output_filename}_parallel_coordinates_plot.html")
    print(f"Plot saved as {output_filename}_parallel_coordinates_plot.html")

def main():
    # Get input filename from user
    input_filename = input("Enter the name of the input file (CSV or TXT): ")
    
    # Determine the delimiter based on file extension
    delimiter = ',' if input_filename.lower().endswith('.csv') else ';'
    
    # Load the data
    df = load_data(input_filename, delimiter)
    
    if df is not None:
        # Add the new column
        df = add_palma_interestingness_bool(df)
        
        # Get output filename from user
        output_filename = input("Enter the name for the output file (without extension): ")
        
        # Create and save the plot
        create_parallel_coordinates_plot(df, output_filename)

if __name__ == "__main__":
    main()
