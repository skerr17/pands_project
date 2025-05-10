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

# import seaborn as sns - Reference: https://seaborn.pydata.org/index.html
import seaborn as sns

# import PCA from sklearn.decomposition - Reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# import StandardScaler from sklearn.preprocessing - Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# Main function to run the analysis
def main():
    """
    Main function to run the analysis using the Iris Data Set.
    """

    # create a directory for the output files
    output_dir = Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

    # Read the data from the iris.data file
    raw_data_dir = Path('inputs') # Path to the iris data file
    raw_data_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist
    data_path = raw_data_dir / 'iris.data' # Path to the iris data file
    # Check if the file exists
    if data_path.exists():
        # Read the data into a pandas dataframe # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        # Add column names to the dataframe from the iris documentation # Reference: https://archive.ics.uci.edu/dataset/53/iris
        iris_data = pd.read_csv(data_path, header=None, usecols=[0, 1, 2, 3, 4], 
                    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    else:
        print(f"Error: The file {data_path} does not exist.")
        return

    # prepare the data for plotting
    variables, variables_titles, species, format_species, colours, labels = prepare_data(iris_data)

    # generate descriptive statistics
    global_iris_descriptive_stats, iris_descriptive_stats_by_species = generate_descriptive_statistics(iris_data, output_dir, variables_titles, species, format_species)

    # print the descriptive statistics to the console
    print('===Global Descriptive Statistics===\n')
    print(tabulate(global_iris_descriptive_stats, headers='keys', tablefmt='grid'))
    print('\n\n')
    print('===Descriptive Statistics by Species===\n')
    print(tabulate(iris_descriptive_stats_by_species.stack(), headers='keys', tablefmt='grid'))

    # plot histograms
    plot_histograms(iris_data, variables, variables_titles, species, colours, labels, output_dir)
    
    # plot scatter plots
    plot_scatter(iris_data, variables, variables_titles, species, colours, labels, output_dir)

    # plot pairs plots
    pairsplots(iris_data, format_species, colours, output_dir)

    # plot correlation matrix heatmap
    corrleation_matrix_heatmap(iris_data, variables_titles, output_dir)

    # perform PCA analysis
    pca_analysis(iris_data, variables, species, colours, output_dir)

    # print a message to indicate that the analysis is complete
    print("Analysis complete: \n" 
            "The following has been produced into the 'outputs' directory:\n"
            "\tThe Descriptive Statistics saved in the 'iris_descriptive_stats.txt',\n"
            "\tHistogram for each Iris Feature (saved in indiviual files),\n"
            "\tScatter plot for each Iris Feature (saved together in the 'iris_scatter.png'),\n"
            "\tPairs plot for each Iris Feature in saved 'iris_pairplot.png',\n"
            "\tCorrelation Matrix Heatmap saved in the 'iris_correlation_matrix.png',\n"
            "\tPCA Analysis saved in the 'iris_pca.png'.\n"
            )
 
    
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
        colours (list): A list of colours for each species.
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

    # Define colours for each species
    colours = ['red', 'green', 'blue'] # Colours for each species
    labels = format_species # Labels for each species

    return variables, variables_titles, species, format_species, colours, labels



def generate_descriptive_statistics(data, output_dir, variables_titles, species, format_species):
    """
    Generate descriptive statistics (Global & by Species) 
    for the given data and writes it to a text file titled
    'iris_descriptive_stats.txt' in the output folder.
    
    Parameters:
    data (DataFrame): The input data to analyze.
    output_dir (Path): The directory to save the output file.
    variables_titles (list): A list of titles for the variables.
    format_species (list): A list of formatted species names.
    
    Returns:
        DataFrame: A DataFrame containing the descriptive statistics.
        Creates a text file with the descriptive statistics.
    """
    # format the data to be more readable
    formatted_data = data.copy()
    formatted_data.columns = variables_titles + ['Species']# rename the columns to be more readable

    # species formatting
    species_mapping = {species[i]: format_species[i] for i in range(len(species))} # create a mapping of species to formatted species names
    formatted_data['Species'] = formatted_data['Species'].replace(species_mapping) # rename the species to be more readable
   
    # Defining the descriptive statistics globally across all species
    # Calculate the descriptive statistics for the iris data
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
    global_descriptive_stats = formatted_data.describe()

    # format global descriptive statistics to be more readable
    global_descriptive_stats = global_descriptive_stats.rename(index={'count': 'Count',
                                                     'mean': 'Mean',
                                                     'std': 'Standard Deviation',
                                                     'min': 'Minimum',
                                                     '25%': '25th Percentile',
                                                     '50%': 'Median',
                                                     '75%': '75th Percentile',
                                                     'max': 'Maximum'})
    # # Print the descriptive statistics for each species
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
    # Reference: https://www.w3schools.com/python/pandas/ref_df_groupby.asp
    stats_by_species =formatted_data.groupby('Species').describe()

    # format the descriptive statistics for each species to be more readable
    stats_by_species = stats_by_species.rename(columns={
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Standard Deviation',
        'min': 'Minimum',
        '25%': '25th Percentile',
        '50%': 'Median',
        '75%': '75th Percentile',
        'max': 'Maximum'},
        )  # Had some issues finding a solution to the formatting of column names but got it to work using the columns parameter of the rename function
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html


    # Write the  Description Stats to a Text File
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html - used to make the data more readable
    with open(output_dir / 'iris_descriptive_stats.txt', 'w') as f:
        # Write the global descriptive statistics to the file
        f.write('===Global Descriptive Statistics===\n')
        f.write(tabulate(global_descriptive_stats, headers='keys', tablefmt='grid'))
        f.write('\n\n')
        # Write the descriptive statistics for each species to the file
        f.write('===Descriptive Statistics by Species===\n')
        f.write(tabulate(stats_by_species.stack(), headers='keys', tablefmt='grid'))
    
    return global_descriptive_stats, stats_by_species


def plot_histograms(data, variables, variables_titles, species, colours, labels, output_dir):
    '''
    Creates the Histograms for each variable with the different species colour coded to individual .png files in the output folder.
    
    Parameters: 
        data (DataFrame): The input data to plot in histograms.
        variables (list): A list of the variables to plot.
        variables_titles (list): A list of titles for the histograms.
        species (list): A list of unique species in the data.
        colours (list): A list of colours for each species.
        labels (list): A list of labels for each species.
        output_dir (Path): The directory to save the output files.
    
    Returns:
        None: The function saves histograms to .png files.

    Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html
    Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    Reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
    '''

    # Loop through each variable and create a subplot
    for i, variable in enumerate(variables):
        # Set the figure size
        plt.figure(figsize=(12, 12))
       
        for j, specie in enumerate(species):
            # Filter the data for the current species
            species_data = data[data['species'] == specie][variable]
            # Plot the histogram for the current species
            plt.hist(species_data, bins=10, alpha=0.7, 
                     color=colours[j], label=labels[j], edgecolor='black')
        
        # Add title, labels, and legend
        plt.title(f'Frequency of {variables_titles[i]} Across Species')
        plt.xlabel(f'{variables_titles[i]} (cm)')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the plot as a PNG file
        filename = output_dir / f'{variable}_histogram.png'
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(filename)
        plt.close() # Close the plot to free up memory
        


def plot_scatter(data, variables, variables_titles, species, colours, labels, output_dir):
    '''
    # Scatter Plot of each pair of variables
    # Reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html

    parameters:
        data (DataFrame): The input data to plot in scatter plots.
        variables (list): A list of the variables to plot.
        variables_titles (list): A list of titles for the histograms.
        species (list): A list of unique species in the data.
        colours (list): A list of colours for each species.
        labels (list): A list of labels for each species.
        output_dir (Path): The directory to save the output files.
    
    returns:
        None: The function saves scatter plots to a single .png file.
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
                            label=labels[k], color=colours[k], alpha=0.7)
            
            # Add title, labels, and legend
            plt.title(f'{variables_titles[i]} vs {variables_titles[j]}')
            plt.xlabel(f'{variables_titles[i]} (cm)')
            plt.ylabel(f'{variables_titles[j]} (cm)')
            plt.legend()
            plot_index += 1

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(output_dir /f'iris_scatter.png')
    plt.close() # Close the plot to free up memory


    
def pairsplots(data, format_species, colours, output_dir):
    '''
    # Pairs Plot of each pair of variables
    # Reference: https://seaborn.pydata.org/generated/seaborn.pairplot.html

    parameters:
        data (DataFrame): The input data to plot in pairs plots.
        format_species (list): A list of formatted species names for the legend.
        colours (list): A list of colours for each species.
        output_dir (Path): The directory to save the output files.
    
    returns:
        None: The function saves pairs plots to a single .png file.
    '''
    # reference: https://seaborn.pydata.org/generated/seaborn.pairplot.html 
    # reference: https://pytutorial.com/python-seaborn-pairplot-visualize-data-relationships/

    iris_pairplot = sns.pairplot(
                                data=data,
                                hue='species', # set the hue to be the species
                                kind='reg', # set the kind of plot to be a regression plot
                                palette=colours, # set the colours of the species
                                diag_kind='kde', # set the diagonal to be a kde plot
                                height=2.5, # set the size of the figure
                                )

    # add the title to the figure
    iris_pairplot.fig.suptitle('Pairplot of the Iris Dataset', y=1.02)

    # Remove the default legend
    iris_pairplot._legend.remove()

    # Create new legend with custom labels
    handles = iris_pairplot._legend_data.values()
    iris_pairplot.fig.legend(
        handles=handles,
        labels=format_species,
        title="Species",
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(format_species)
    )

    # save the plot as a PNG file
    plt.savefig(output_dir / 'iris_pairplot.png')
    plt.close() # Close the plot to free up memory



def corrleation_matrix_heatmap(data, variables_titles, output_dir):
    '''
    # Correlation Matrix Heatmap of the Iris Dataset
    # Reference: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # Reference: https://www.geeksforgeeks.org/python-pandas-dataframe-corr/


    parameters:
        data (DataFrame): The input data to plot in heatmaps.
        variables_titles (list): A list of titles for the histograms.
        output_dir (Path): The directory to save the output files.
    
    returns:
        None: The function saves heatmaps to a single .png file.
    '''

    # calculate the correlation matrix and drops the species column as it is not a numeric column
    corr_matrix = data.drop(columns=['species']).corr()

    # set up the matplotlib figure
    plt.figure(figsize=(12, 12))

    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, 
                cmap='coolwarm', 
                annot=True, 
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8}, 
                linewidths=.5,
                xticklabels=variables_titles,
                yticklabels=variables_titles,)

    # add title to the figure
    plt.title('Correlation Matrix Heatmap of the Iris Dataset')

    # save the plot as a PNG file
    plt.savefig(output_dir / 'iris_correlation_matrix.png')
    plt.close() # Close the plot to free up memory



def pca_analysis(data, variables, species, colours, output_dir):
    '''
    # PCA Analysis of the Iris Dataset 
    - Purpose of PCA is to reduce the dimensionality of the data while preserving as much variance as possible.
    - PCA is a technique used to identify patterns in data and express the data in such a way as to highlight their similarities and differences.
    - For the Iris dataset, PCA is used to reduce the 4D data (sepal length, sepal width, petal length, petal width) to 2D data (PC1 and PC2). 
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # Reference: https://www.geeksforgeeks.org/principal-component-analysis-pca/

    parameters:
        data (DataFrame): The input data to plot in heatmaps.
        variables (list): A list of the variables to plot.
        species (list): A list of unique species in the data.
        colours (list): A list of colours for each species.
        output_dir (Path): The directory to save the output files.
    
    returns:
        None: The function saves a PCA Scatter Plot to a single .png file.
    '''


    # standardize the data
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    # fit the scaler to the data and transform the data
    scaled_data = scaler.fit_transform(data[variables])

    # perform PCA
    pca = PCA(n_components=2) # set the number of components to 2
    # fit the PCA model to the scaled data
    pca_result = pca.fit_transform(scaled_data) 

    # create Dataframe for PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2']) # create a DataFrame for the PCA results
    pca_df['species'] = data['species'] # add the species column to the PCA results

    # plot the PCA results
    plt.figure(figsize=(12, 12)) # set the figure size
    
    # create a scatter plot of the PCA results
    plt.scatter(data=pca_df, 
                x='PC1', 
                y='PC2', 
                c=pca_df['species'].map(dict(zip(species, colours))) # map the species to the colours had to create a dict to map the species to the colours
                ) 
    plt.title('PCA of the Iris Dataset') # add title to the figure
    plt.xlabel('Principal Component 1') # add x label to the figure
    plt.ylabel('Principal Component 2') # add y label to the figure
    plt.legend(title='Species') # add legend to the figure
    plt.savefig(output_dir / 'iris_pca.png') # save the plot as a PNG file
    plt.close() # Close the plot to free up memory


if __name__ == "__main__":
    # main function to run the analysis
    main()