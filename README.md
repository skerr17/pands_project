# **Programming & Scripting Project**

By Stephen Kerr

## Purpose of this Repository 
In this repository, I will submit my main project assessment for the Module Programming & Scripting taught by Andrew Beatty (email: andrew.beatty@atu.ie) for my HDIP in Computing and Data Analytics.

The following Problem Statement was assigned: 
>'This project concerns the well-known Fisher’s Iris data set. You must research the data set
and write documentation and code (in Python) to investigate it. An online search for
information on the data set will convince you that many people have investigated it
previously. You are expected to be able to break this project into several smaller tasks that
are easier to solve, and to plug these together after they have been completed.'

---

## The Iris Dataset
The **Iris Dataset** contains 150 samples of Iris flowers from three species: **Setosa**, **Versicolor**, and **Virginica**.

Each samples includes four features: 
- **Sepal length** (in cm)
- **Sepal width** (in cm)
- **Petal length** (in cm)
- **Petal width** (in cm)

The raw data can be seen in [iris.data](https://github.com/skerr17/pands_project/blob/main/iris.data) which was sourced from [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris). 

The Image below illustrates the Three Iris Flower Species and their anatomy (Image sourced from [here](https://www.analyticsvidhya.com/blog/2022/06/iris-flowers-classification-using-machine-learning/)).

"![iris flower image](https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png)


## Technologies Used 

- Python 3.12.7 packaged by [Anaconda](https://www.anaconda.com/download)
- [Visual Studio Code](https://visualstudio.microsoft.com/)
- Git & Github

## Project Plan
1. Researched the Iris dataset.
2. Loaded and explored the dataset using python (and the python libraries listed in the [requirements.txt](https://github.com/skerr17/pands_project/blob/main/requirements.txt)).
3. Write code to compute basic statistics.
4. Created histograms to visualise the distribution of each variable.
5. Generated scatter plots to study the relationships between variable pairs.
6. Created a pair plot to visualize relationships across all numeric features.
7. Generated a Correlation Matrix Heatmap of the different features (Pearson Correlation Coefficient).
8. Conducted Principal Component Analysis (PCA) to turn the 4 dimensional Iris Dataset to 2 dimensional with a total explained variance of 95.80%.
9. Documented the analysis and committed each logical change to GitHub.
10. Documented the analysis, insights and observations of the Iris dataset in to a Juypeter Notebook.



## Repository Contents 


- [inputs folder](https://github.com/skerr17/pands_project/tree/main/inputs) - This folder contains the following inputs to the project: 
    - [iris.data](https://github.com/skerr17/pands_project/blob/main/iris.data) - This file contains the famous Iris Data that was sourced from - [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris).
    - [iris_species_image.png](https://github.com/skerr17/pands_project/blob/main/inputs/iris_species_image.png). Image sourced from [here](https://www.analyticsvidhya.com/blog/2022/06/iris-flowers-classification-using-machine-learning/).


- [outputs folder](https://github.com/skerr17/pands_project/tree/main/outputs) - This folder contains all the outputs of the analysis of this project: 
    - [iris_descriptive_stats.txt](outputs/iris_descriptive_stats.txt)
    - [petal_length_histogram.png](https://github.com/skerr17/pands_project/blob/main/outputs/petal_length_histogram.png)
    - [petal_width_histogram.png](https://github.com/skerr17/pands_project/blob/main/outputs/petal_width_histogram.png)
    - [sepal_length_histogram.png](https://github.com/skerr17/pands_project/blob/main/outputs/sepal_length_histogram.png)
    - [sepal_width_histogram.png](https://github.com/skerr17/pands_project/blob/main/outputs/sepal_width_histogram.png)
    - [iris_scatter.png](https://github.com/skerr17/pands_project/blob/main/outputs/iris_scatter.png)
    - [iris_pairplot.png](https://github.com/skerr17/pands_project/blob/main/outputs/iris_pairplot.png)
    - [iris_correlation_matrix.png](https://github.com/skerr17/pands_project/blob/main/outputs/iris_correlation_matrix.png)
    - [iris_pca.png](https://github.com/skerr17/pands_project/blob/main/outputs/iris_pca.png)


- [analysis_code.py](https://github.com/skerr17/pands_project/blob/main/analysis_code.py) - This program contains the code for main analysis of the Iris Dataset.
- [iris_dataset_analysis.ipynb](https://github.com/skerr17/pands_project/blob/main/analysis_code.py) - This Jupyter Notebook contains the observations and comments on the analysis of the Iris Dataset.
- [requirements.txt](https://github.com/skerr17/pands_project/blob/main/requirements.txt) - This file contains the dependencies required to run the code in this repository. 