# analysis.py
# This program is the central analysis file for the Programming and Scripting Module main project.
# It imports the Iris Data and perfoms the analysis on it.
# Author: Stephen Kerr

# import pandas as pd
import pandas as pd

# import numpy as np
import numpy as np

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# import tabulate - For nice tables - Reference: https://pypi.org/project/tabulate/
from tabulate import tabulate

# from pathlib import Path - Reference: https://docs.python.org/3/library/pathlib.html
# imported pathlib as it has better error handling and is more robust and OS independent than os.path
from pathlib import Path


def generate_descriptive_statistics(data):
    """
    Generate descriptive statistics (Global & by Species) 
    for the given data and writes it to a text file titled
    'iris_descriptive_stats.txt'.
    
    Parameters:
    data (DataFrame): The input data to analyze.
    
    Returns:
        DataFrame: A DataFrame containing the descriptive statistics.
        Creates a text file with the descriptive statistics.
    """
    # Defining the descriptive statistics globally across all species
    # Calculate the descriptive statistics for the iris data
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
    global_descriptive_stats = data.describe()


    # # Print the descriptive statistics for each species
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
    # Reference: https://www.w3schools.com/python/pandas/ref_df_groupby.asp
    stats_by_species =data.groupby('species').describe()


    # Write the  Description Stats to a Text File
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html - used to make the data more readable
    with open('iris_descriptive_stats.txt', 'w') as f:
        # Write the global descriptive statistics to the file
        f.write('===Global Descriptive Statistics===\n')
        f.write(tabulate(global_descriptive_stats, headers='keys', tablefmt='grid'))
        f.write('\n\n')
        # Write the descriptive statistics for each species to the file
        f.write('===Descriptive Statistics by Species===\n')
        f.write(tabulate(stats_by_species.stack(), headers='keys', tablefmt='grid'))
    
    return global_descriptive_stats, stats_by_species


def prepare_data(data):
    """
    Prepares the data for the analysis plots. 

    Parameters:
        data (DataFrame): The input data to prepare.
    
    Returns:
        variables (list): A list of the variables to plot.
        variables_titles (list): A list of titles for the histograms.
        species (list): A list of unique species in the data.
        format_species (list): A list of formatted species names for the legend.
        colors (list): A list of colors for each species.
        labels (list): A list of labels for each species.
    """

    # Create a list of the variables to plot
    variables = data.columns.drop('species') # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # Create a list of titles for the histograms
    variables_titles = [s.replace('_', ' ').title() for s in variables] # ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    # Get the unique species from the data
    species = data['species'].unique() # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Format the species names for the legend
    format_species = [s.replace('Iris-', '').capitalize() for s in species] # ['Setosa', 'Versicolor', 'Virginica']

    # Define colors for each species
    colors = ['red', 'green', 'blue'] # Colors for each species
    labels = format_species # Labels for each species

    return variables, variables_titles, species, format_species, colors, labels



def plot_histograms(data, variables, variables_titles, species, format_species, colors, labels):
    '''
    Creates the Histograms for each variable with the different species colour coded to a .png file
    
    Parameters: 
        data (DataFrame): The input data to plot in histograms.
        variables (list): A list of the variables to plot.
        variables_titles (list): A list of titles for the histograms.
        species (list): A list of unique species in the data.
        format_species (list): A list of formatted species names for the legend.
        colors (list): A list of colors for each species.
        labels (list): A list of labels for each species.
    
    Returns:
        None: The function saves the histograms as a .png file.

    Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
    Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
    '''


    # Set the figure size
    plt.figure(figsize=(12, 12))

    # plot index
    plot_index = 1

    # Loop through each variable and create a subplot
    for i, variable in enumerate(variables):
        plt.subplot(2, 2, i + 1)  # Create a 2x2 grid of subplots
        for j, specie in enumerate(species):
            # Filter the data for the current species
            species_data = data[data['species'] == specie][variable]
            # Plot the histogram for the current species
            plt.hist(species_data, bins=10, alpha=0.7, 
                     color=colors[j], label=labels[j], edgecolor='black')
        
        # Add title, labels, and legend
        plt.title(f'Frequency of {variables_titles[i]} Across Species')
        plt.xlabel(f'{variables_titles[i]} (cm)')
        plt.ylabel('Frequency')
        plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('iris_histograms.png')


def plot_scatter(data, variables, variables_titles, species, format_species, colors, labels):
    '''
    # Scatter Plot of each pair of variables
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html
    '''
    
    # set the figure size
    plt.figure(figsize=(12, 12))

    # plot index 
    plot_index = 1

    # Loop through each variable and create a subplot
    for i, variable_1 in enumerate(variables):
        for j, variable_2 in enumerate(variables):
            if i >= j:
                continue
            plt.subplot(len(variables), len(variables), plot_index) # Create a grid of subplots
            for k, specie in enumerate(species):
                # Filter the data for the current species
                species_data = data[data['species'] == specie]
                # Plot the scatter plot for the current species
                plt.scatter(species_data[variable_1], species_data[variable_2], 
                            label=labels[k], color=colors[k], alpha=0.7)
            
            # Add title, labels, and legend
            plt.title(f'{variable_1} vs {variable_2}')
            plt.xlabel(f'{variable_1} (cm)')
            plt.ylabel(f'{variable_2} (cm)')
            plt.legend()
            plot_index += 1

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('iris_scatter.png')

    

# Main function to run the analysis
def main():
    """
    Main function to run the analysis using the Iris Data Set.
    """
    # Read the data from the iris.data file
    data_path = Path('iris.data') # Path to the iris data file
    # Check if the file exists
    if data_path.exists():
        # Read the data into a pandas dataframe # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        # Add column names to the dataframe from the iris documentation # Reference: https://archive.ics.uci.edu/dataset/53/iris
        iris_data = pd.read_csv(data_path, header=None, usecols=[0, 1, 2, 3, 4], 
                    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    else:
        print(f"Error: The file {data_path} does not exist.")
        return
            
    # generate descriptive statistics
    global_iris_descriptive_stats, iris_descriptive_stats_by_species = generate_descriptive_statistics(iris_data)

    # prepare the data for plotting
    variables, variables_titles, species, format_species, colors, labels = prepare_data(iris_data)

    # print the descriptive statistics to the console
    print('===Global Descriptive Statistics===\n')
    print(tabulate(global_iris_descriptive_stats, headers='keys', tablefmt='grid'))
    print('\n\n')
    print('===Descriptive Statistics by Species===\n')
    print(tabulate(iris_descriptive_stats_by_species.stack(), headers='keys', tablefmt='grid'))

    # plot histograms
    plot_histograms(iris_data, variables, variables_titles, species, format_species, colors, labels)
    
    # plot scatter plots
    plot_scatter(iris_data, variables, variables_titles, species, format_species, colors, labels)

    # show the plots
    plt.show()

    # print a message to indicate that the analysis is complete
    print("Analysis complete: \n" 
            "The histograms are saved to the iris_histograms.png\n"
            "The scatter plots are saved to the iris_scatter.png\n"
            "The descriptive statistics are saved to the iris_descriptive_stats.txt\n")


 
    
if __name__ == "__main__":
    # main function to run the analysis
    main()