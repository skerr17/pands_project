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
    f.write(tabulate(iris_descriptive_stats_by_species.stack(), headers='keys', tablefmt='grid'))

