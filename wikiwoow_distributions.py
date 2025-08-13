#!/usr/bin/env python3
"""
Excel Distribution Analyzer
Generates distribution curves from Excel data columns
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import argparse
import sys
import os


def analyze_distribution(file_path, column_name, sheet_name=None, output_dir=None):
    """
    Analyze and plot distribution of values from an Excel or CSV file column

    Args:
        file_path (str): Path to Excel (.xlsx, .xls) or CSV (.csv) file
        column_name (str): Name of the column to analyze
        sheet_name (str): Sheet name (optional, only for Excel files)
        output_dir (str): Directory to save plots (optional)
    """

    try:
        # Determine file type and read accordingly
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv':
            print(f"Reading CSV file: {file_path}")
            # Try multiple CSV reading strategies
            try:
                # First attempt: standard reading
                df = pd.read_csv(file_path)

                # Check if we have a semicolon-separated file (all data in one column)
                if len(df.columns) == 1 and ';' in df.columns[0]:
                    print("Detected semicolon-separated file, re-reading with semicolon delimiter...")
                    df = pd.read_csv(file_path, sep=';')

            except pd.errors.ParserError as e:
                print(f"Standard CSV parsing failed: {e}")
                print("Trying alternative parsing methods...")

                try:
                    # Second attempt: try semicolon separator first
                    df = pd.read_csv(file_path, sep=';', engine='python')
                    print("Successfully parsed with semicolon separator")
                except:
                    try:
                        # Third attempt: more flexible parsing
                        df = pd.read_csv(file_path,
                                         sep=None,  # Auto-detect separator
                                         engine='python',  # More flexible parser
                                         encoding='utf-8',
                                         on_bad_lines='skip')  # Skip bad lines
                        print("Successfully parsed with auto-detected separator")
                    except:
                        try:
                            # Fourth attempt: handle inconsistent columns
                            df = pd.read_csv(file_path,
                                             sep=',',
                                             engine='python',
                                             encoding='utf-8',
                                             on_bad_lines='skip',  # Skip problematic lines
                                             quoting=1)  # Handle quotes properly
                            print("Successfully parsed by skipping bad lines")
                        except Exception as final_error:
                            print(f"All CSV parsing attempts failed. Final error: {final_error}")
                            print("\nTroubleshooting suggestions:")
                            print("1. Check if the file has mixed delimiters (commas, semicolons, tabs)")
                            print("2. Look for unescaped quotes or special characters")
                            print("3. Check if some rows have extra commas")
                            print("4. Try opening the file in a text editor to examine the structure")
                            return

            # Clean up empty columns that might have been created
            df = df.dropna(axis=1, how='all')  # Remove completely empty columns
            df.columns = df.columns.str.strip()  # Remove whitespace from column names

            if sheet_name:
                print("Warning: Sheet name ignored for CSV files")
        elif file_extension in ['.xlsx', '.xls']:
            print(f"Reading Excel file: {file_path}")
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
        else:
            print(f"Error: Unsupported file format '{file_extension}'. Please use .csv, .xlsx, or .xls files.")
            return

        print(f"Available columns: {list(df.columns)}")

        # Check if column exists
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found!")
            print(f"Available columns: {list(df.columns)}")
            return

        # Extract and clean data
        data = df[column_name].dropna()

        if len(data) == 0:
            print(f"Error: No valid data found in column '{column_name}'")
            return

        # Convert to numeric if possible
        try:
            data = pd.to_numeric(data, errors='coerce').dropna()
        except:
            print(f"Warning: Some values in '{column_name}' are not numeric")

        if len(data) == 0:
            print(f"Error: No numeric data found in column '{column_name}'")
            return

        # Calculate statistics
        stats_dict = {
            'Count': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Mode': data.mode().iloc[0] if not data.mode().empty else 'N/A',
            'Std Dev': data.std(),
            'Variance': data.var(),
            'Min': data.min(),
            'Max': data.max(),
            'Range': data.max() - data.min(),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data)
        }

        # Print statistics
        print("\n" + "=" * 50)
        print(f"STATISTICAL SUMMARY FOR COLUMN: '{column_name}'")
        print("=" * 50)
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)) and key != 'Count':
                print(f"{key:12}: {value:.4f}")
            else:
                print(f"{key:12}: {value}")
        print("=" * 50)

        # Create the plots
        create_distribution_plots(data, column_name, stats_dict, file_path, output_dir)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
    except Exception as e:
        print(f"Error: {str(e)}")


def create_distribution_plots(data, column_name, stats_dict, file_path, output_dir=None):
    """Create comprehensive distribution plots"""

    # Set style
    plt.style.use('seaborn-v0_8')

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Main title
    fig.suptitle(f'Distribution Analysis: {column_name}\nFile: {os.path.basename(file_path)}',
                 fontsize=16, fontweight='bold')

    # 1. Histogram with KDE
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(data, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')

    # Add KDE curve
    x_range = np.linspace(data.min(), data.max(), 100)
    kde = stats.gaussian_kde(data)
    plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add normal distribution overlay
    mu, sigma = data.mean(), data.std()
    normal_curve = stats.norm.pdf(x_range, mu, sigma)
    plt.plot(x_range, normal_curve, 'g--', linewidth=2, label='Normal Fit')

    # Add mean line
    plt.axvline(mu, color='blue', linestyle=':', linewidth=2, label=f'Mean: {mu:.2f}')

    plt.title('Histogram with Distribution Curves')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Box Plot
    ax2 = plt.subplot(2, 3, 2)
    box_plot = plt.boxplot(data, vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    plt.title('Box Plot')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3)

    # 3. Q-Q Plot (Normal)
    ax3 = plt.subplot(2, 3, 3)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal Distribution)')
    plt.grid(True, alpha=0.3)

    # 4. Violin Plot
    ax4 = plt.subplot(2, 3, 4)
    parts = plt.violinplot([data], positions=[1], widths=0.5)
    parts['bodies'][0].set_facecolor('lightgreen')
    plt.title('Violin Plot')
    plt.ylabel('Values')
    plt.xticks([1], [column_name])
    plt.grid(True, alpha=0.3)

    # 5. Empirical CDF
    ax5 = plt.subplot(2, 3, 5)
    sorted_data = np.sort(data)
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y_values, 'b-', linewidth=2)
    plt.title('Empirical Cumulative Distribution')
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)

    # 6. Statistics Text Box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
    STATISTICS SUMMARY
    ==================
    Count:     {stats_dict['Count']:,}
    Mean:      {stats_dict['Mean']:.4f}
    Median:    {stats_dict['Median']:.4f}
    Std Dev:   {stats_dict['Std Dev']:.4f}
    Variance:  {stats_dict['Variance']:.4f}

    Min:       {stats_dict['Min']:.4f}
    Max:       {stats_dict['Max']:.4f}
    Range:     {stats_dict['Range']:.4f}

    Skewness:  {stats_dict['Skewness']:.4f}
    Kurtosis:  {stats_dict['Kurtosis']:.4f}
    """

    plt.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save plot if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"distribution_{column_name.replace(' ', '_').replace('/', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {filepath}")

    plt.show()


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Generate distribution curves from Excel or CSV data')
    parser.add_argument('file_path', help='Path to Excel (.xlsx, .xls) or CSV (.csv) file')
    parser.add_argument('column_name', help='Name of column to analyze')
    parser.add_argument('--sheet', help='Sheet name (optional, Excel files only)')
    parser.add_argument('--output', help='Output directory for saving plots')

    args = parser.parse_args()

    analyze_distribution(args.file_path, args.column_name, args.sheet, args.output)


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("Data Distribution Analyzer")
        print("=" * 30)
        print("\nSupported formats: Excel (.xlsx, .xls) and CSV (.csv)")
        print("\nUsage:")
        print("python script.py <file_path> <column_name> [--sheet <sheet_name>] [--output <output_dir>]")
        print("\nExamples:")
        print("python script.py data.xlsx 'Sales Amount' --sheet 'Sheet1' --output ./plots")
        print("python script.py data.csv 'Temperature' --output ./plots")
        print("\nOr use interactively:")

        file_path = input("Enter file path (Excel or CSV): ").strip()
        column_name = input("Enter column name: ").strip()
        sheet_name = input("Enter sheet name (Excel only, or press Enter for default): ").strip() or None
        output_dir = input("Enter output directory (or press Enter to skip): ").strip() or None

        if file_path and column_name:
            analyze_distribution(file_path, column_name, sheet_name, output_dir)
    else:
        main()