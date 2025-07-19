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
        'Interestingness': ['InterestingnessEnt1Ent2', 'Interestingness'],
        'Interestingness2': ['Interestingness2Ent1Ent2', 'Interestingness2'],
        'Interestingness3': ['Interestingness3Ent1Ent2', 'Interestingness3'],
        'Interestingness4': ['Interestingness4Ent1Ent2', 'Interestingness4'],
        'Interestingness5': ['Interestingness5Ent1Ent2', 'Interestingness5'],
        'Perceived_Relatedness': ['Perceived_Relatedness', 'Perceived_Relatedness'],
        'Subjective_Knowledge': ['Subjective_Knowledge', 'Subjective_Knowledge'],
        'Serendipity': ['Serendipity', 'Serendipity']
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
    delimiter = ';' if input_filename.lower().endswith('.csv') else ','
    
    # Load the data
    df = load_data(input_filename, delimiter)
    
    if df is not None:
        # Get output filename from user
        output_filename = input("Enter the name for the output file (without extension): ")
        
        # Create and save the plot
        create_parallel_coordinates_plot(df, output_filename)

if __name__ == "__main__":
    main()