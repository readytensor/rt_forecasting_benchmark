import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chardet

# Load the benchmarking results
file_path = './../inputs/Forecasting results 20240221.csv'
heatmap_file_path = './../outputs/forecasting_models_heatmap.png'


with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())
encoding = result['encoding']
df = pd.read_csv(file_path, encoding=encoding)

# Set 'model_name' as the index to use it as row labels
data_models_only_index = df.set_index('model_name')

# Select only the numeric data for visualization
numeric_data_models_only = data_models_only_index.select_dtypes(include='number')

# Creating the heatmap
plt.figure(figsize=(12, 14))
ax = sns.heatmap(numeric_data_models_only, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, 
                 vmin=numeric_data_models_only.min().min(), vmax=numeric_data_models_only.max().max())

# Title and adjustments
plt.title('Performance Heat Map of Forecasting Models Using RMSSE Score', fontsize=16)
plt.xticks(rotation=45)
plt.ylabel('')  # Remove the x-axis label
plt.tight_layout()

# Defining the boxes' start and end rows (Python uses 0-based indexing)
boxes = [(0, 4), (4, 9), (9, 11), (11, 23), (23, 42)]

# Adding borders around specified groups of models
for start, end in boxes:
    ax.add_patch(
        plt.Rectangle(
            (0, start), 
            numeric_data_models_only.shape[1],
            end-start,
            fill=False,
            edgecolor='black',
            lw=2,
            clip_on=False
        )
    )

# Save the figure
plt.savefig(heatmap_file_path)
plt.close()  # Close the plot to prevent it from displaying again
