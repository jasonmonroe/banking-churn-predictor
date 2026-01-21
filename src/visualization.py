import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
import src.constants as const

""" 
Perform Univariate Analysis by creating box plots, histograms, density plots for each column.
"""

def observe_data(data: pd.DataFrame):
    print(data.head())
    print(data.tail())
    print(data.shape)
    print(data.info())
    print(data.describe().T)

    # Checking the dtypes of the variables in the data
    print(data.dtypes)

    # Find any missing values
    data.isnull().sum()


def show_visualizations(data: pd.DataFrame):
    # Display histogram
    histogram_boxplot(data, 'credit_score')

    # Display Geography barplot
    labeled_barplot(data, 'geography', perc=True)

    # Display Gender barplot
    labeled_barplot(data, 'gender', perc=True)

    # Display age histogram
    histogram_boxplot(data, 'age')

    # Display tenure barplot
    labeled_barplot(data, 'tenure', perc=True)

    labeled_barplot(data, 'num_of_products', perc=True)

    histogram_boxplot(data, 'balance')

    labeled_barplot(data, 'has_cr_card', perc=True)

    histogram_boxplot(data, 'estimated_salary')

    labeled_barplot(data, 'is_active_member', perc=True)

    # Display exited barplot
    labeled_barplot(data, 'exited', perc=True)


def show_salary_barplot_visualization(df: pd.DataFrame):
    # Compare estimated salary to bank balance to see if there's a correlation.
    # Note: Group the balance by $20,000s to lower the number of values on the x-axis
    barplot_data = df.copy()
    barplot_data['estimated_salary'] = barplot_data['estimated_salary'].apply(lambda x: x // const.BALANCE_THRESHOLD)
    barplot_data['balance'] = barplot_data['balance'].apply(lambda x: x // const.BALANCE_THRESHOLD)

    # Call stacked barplot with the modified DataFrame
    stacked_barplot(barplot_data, 'estimated_salary', 'balance')


def show_plot_distributions(df: pd.DataFrame):
    """
    # Create heatmaps to compare two columns: scatter plots, correlation coefficients, cross-tabulation, pair plot, et
    :param df:
    :return:
    """
    data_columns = [
        'estimated_salary',
        'balance',
        'age',
        'gender',
        'tenure',
        'is_active_member',
        'num_of_products',
        'credit_score',
        'geography',
        'has_cr_card',
    ]

    # Compare column to whether they (customer) exited (the program).
    for col_name in data_columns:
        distribution_plot_wrt_target(df, col_name, 'exited')

    # Compare Number of Products for customers with a stacked barplot.
    stacked_barplot(df, 'num_of_products', 'exited')


def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.
    """
    # Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.show()

# Define plot to determine model performance
def plot_model_performance(mod_hist: History, label: str, title: str = '') -> None:

    """
    Function to plot loss/accuracy

    mod_hist: an object which stores the metrics and losses.
    label: can be one of Loss or Accuracy
    """

    fig, ax = plt.subplots() # Creating a subplot with a figure and axes.
    plt.plot(mod_hist.history[label]) # Plotting the train accuracy or train loss
    plt.plot(mod_hist.history['val_'+label.lower()]) # Plotting the validation accuracy or validation loss

    plt.title(f'{title.title()} Model: {label.title()}') # Defining the title of the plot.
    plt.ylabel(label.capitalize()) # Capitalizing the first letter.
    plt.xlabel('Epochs') # Defining the label for the x-axis.
    fig.legend(['Train', 'Validation'], loc="outside right upper") # Defining the legend, loc controls the position of the legend.

# Define labeled barplot.
def labeled_barplot(data: pd.DataFrame, feature:str, perc:bool=False, n=None) -> None:
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count
    n: displays the top n category levels
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()

    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )
        else:
            label = p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.show()  # to avoid overlap

# Define a histogram boxplot
def histogram_boxplot(data: pd.DataFrame, feature: str, figsize: tuple=(12, 7), kde: bool=False, bins=None) -> None:
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,  # creating
    )

    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )

    # boxplot will be created and a star will indicate the mean value
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )

    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )

    # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )

    # Add median to the histogram
    plt.show()

# Define stacked barplot
def stacked_barplot(data: pd.DataFrame, predictor: str, target: str) -> None:
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )

    print(tab1)
    print("-" * 120)

    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )

    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

# Plot distributions
def distribution_plot_wrt_target(data: pd.DataFrame, predictor: str, target: str) -> None:

    # Create heatmaps to compare two columns: scatter plots, correlation coefficients, cross-tabulation, pair plot, etc
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


def show_correlation_matrix(df: pd.DataFrame):

    # Exclude columns row_number, customer_id and surname as they are not needed for the matrix.
    corr_data = df.copy()

    for col in corr_data.columns:
        new_col_name = col.replace('_', ' ').title()
        corr_data[new_col_name] = corr_data[col]
        corr_data.drop(columns=col, inplace=True)

    plt.figure(figsize=(15, 7))
    sns.heatmap(corr_data.corr(numeric_only = True), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap='Spectral')
    plt.title('Correlation Matrix of Bank Customer Churn')
    plt.show()
