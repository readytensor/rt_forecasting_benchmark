import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
import sys

# Load the benchmarking results
file_path = './../outputs/aggregate-scores.csv'
heatmap_file_path = './../outputs/forecasting_models_heatmap.png'

def read_data(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    encoding = result['encoding']
    df = pd.read_csv(file_path, encoding=encoding)
    return df


def create_and_save_heatmap_chart(df):

    # Calculate start and end index for each group
    group_bounds = []
    for group_name, group_df in df.groupby('category'):
        start_index = group_df.index.min()
        end_index = group_df.index.max()
        group_bounds.append((start_index, end_index))
    sorted_group_bounds = sorted(group_bounds, key=lambda x: x[0])


    # Set 'model_name' as the index to use it as row labels
    data_models_only_index = df.drop('category', axis=1).set_index('model_name')

    # Select only the numeric data for visualization
    numeric_data_models_only = data_models_only_index.select_dtypes(include='number')

    # Creating the heatmap
    plt.figure(figsize=(12, 16))
    ax = sns.heatmap(numeric_data_models_only, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, 
                    vmin=numeric_data_models_only.min().min(), vmax=numeric_data_models_only.max().max())

    # Title and adjustments
    plt.title('Performance Heat Map of Forecasting Models Using RMSSE Score', fontsize=16)
    plt.xticks(rotation=45)
    plt.ylabel('')  # Remove the x-axis label
    plt.tight_layout()

    # Defining the boxes' start and end rows (Python uses 0-based indexing)
    # Adding borders around specified groups of models
    for start, end in sorted_group_bounds:
        ax.add_patch(
            plt.Rectangle(
                (0, start), 
                numeric_data_models_only.shape[1],
                end-start+1,
                fill=False,
                edgecolor='black',
                lw=2,
                clip_on=False
            )
        )

    # Save the figure
    plt.savefig(heatmap_file_path)
    plt.close()  # Close the plot to prevent it from displaying again


def create_heatmap_chart():
    df = read_data(file_path)
    create_and_save_heatmap_chart(df)


if __name__ == '__main__':
    create_heatmap_chart()