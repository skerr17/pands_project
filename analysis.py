# This program is the central analysis file for the Programming and Scripting Module.
# It imports the Iris Data and perfoms the analysis on it.
# Author: Stephen Kerr

# import pandas as pd
import pandas as pd

# import numpy as np
import numpy as np

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# import tabulate - For nice tables 
# # Reference: https://pypi.org/project/tabulate/
from tabulate import tabulate 



# Read the data into a pandas dataframe # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# Add column names to the dataframe from the iris documentation # Reference: https://archive.ics.uci.edu/dataset/53/iris
iris_data = pd.read_csv('iris.data', header=None, usecols=[0, 1, 2, 3, 4], 
                        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])


# Defining the descriptive statistics globally across all species
# Calculate the descriptive statistics for the iris data
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
global_iris_descriptive_stats = iris_data.describe()

# # Print the descriptive statistics for each species
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
# Reference: https://www.w3schools.com/python/pandas/ref_df_groupby.asp
iris_descriptive_stats_by_species =iris_data.groupby('species').describe()

# Write the  Description Stats to a Text File
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html - used to make the data more readable
with open('iris_descriptive_stats.txt', 'w') as f:
    # Write the global descriptive statistics to the file
    f.write('===Global Descriptive Statistics===\n')
    f.write(tabulate(global_iris_descriptive_stats, headers='keys', tablefmt='grid'))
    f.write('\n\n')
    # Write the descriptive statistics for each species to the file
    f.write('===Descriptive Statistics by Species===\n')
    f.write(tabulate(iris_descriptive_stats_by_species.stack(future_stack=True), headers='keys', tablefmt='grid'))


# Histograms for each variable to a .png file
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
# Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html



# Create a list of the variables to plot
iris_variables = iris_data.columns.drop('species') # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Create a list of titles for the histograms
iris_variables_titles = [s.replace('_', ' ').title() for s in iris_variables] # ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# Get the unique species from the data
species = iris_data['species'].unique() # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Format the species names for the legend
format_species = [s.replace('Iris-', '').capitalize() for s in species] # ['Setosa', 'Versicolor', 'Virginica']

# Define colors for each species
colors = ['red', 'green', 'blue'] # Colors for each species
labels = format_species # Labels for each species

# Set the figure size
plt.figure(figsize=(12, 12))

# Loop through each variable and create a subplot
for i, variable in enumerate(iris_variables):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid of subplots
    for j, specie in enumerate(species):
        # Filter the data for the current species
        species_data = iris_data[iris_data['species'] == specie][variable]
        # Plot the histogram for the current species
        plt.hist(species_data, bins=10, alpha=0.7, color=colors[j], label=labels[j], edgecolor='black')
    
    # Add title, labels, and legend
    plt.title(f'Frequency of {iris_variables_titles[i]} Across Species')
    plt.xlabel(f'{iris_variables_titles[i]} (cm)')
    plt.ylabel('Frequency')
    plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('iris_histograms.png')



# Scatter Plot of each pair of variables
# Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html

# set the figure size
plt.figure(figsize=(12, 12))

# Note reuse the vaiables defined earlier 

# Loop through each variable and create a subplot
for i, variable in enumerate(iris_variables):
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid of subplots
    for j, specie in enumerate(species):
        # Filter the data for the current species
        species_data = iris_data[iris_data['species'] == specie]
        # Plot the scatter plot for the current species
        plt.scatter(species_data[variable], species_data[iris_variables[i - 1]], 
                    label=labels[j], color=colors[j], alpha=0.7)
    
    # Add title, labels, and legend
    plt.title(f'{iris_variables_titles[i]} vs {iris_variables_titles[i - 1]}')
    plt.xlabel(f'{iris_variables_titles[i]} (cm)')
    plt.ylabel(f'{iris_variables_titles[i - 1]} (cm)')
    plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('iris_scatter.png')


# show the plot
plt.show()


