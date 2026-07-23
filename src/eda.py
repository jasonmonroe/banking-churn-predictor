"""
Module for Exploratory Data Analysis (EDA) and visualization functions.

This module provides a collection of functions to perform univariate and
bivariate analysis, visualize data distributions, create various plots
(histograms, box plots, bar plots, correlation matrices), and display
model performance metrics such as classification reports.
"""

# Standard Library Imports
from typing import List, Optional, Tuple

# Third-party Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Sequential

# Local Imports
from src.constants import (
    BALANCE_THRESHOLD,
    CUSTOMER_CHURN_PROB_THRESHOLD,
    PEP8_LINE_LEN,
    TARGET_COL,
)
from src.utils import show_banner

# Set matplotlib backend to 'Agg' to prevent display issues in non-GUI environments
matplotlib.use("Agg")


def show_visualizations(data: pd.DataFrame) -> None:
    """
    Displays a series of visualizations for key features in the dataset.

    This includes histogram-boxplots for numerical features and labeled
    bar plots for categorical features.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data
                             to visualize.
    """
    print("# --- 📊 Showing Visualizations 📊 --- #")

    # Display histogram and boxplot for 'credit_score'
    histogram_boxplot(data, "credit_score")

    # Display labeled bar plot for 'geography'
    labeled_barplot(data, "geography", perc=True)

    # Display labeled bar plot for 'gender'
    labeled_barplot(data, "gender", perc=True)

    # Display histogram and boxplot for 'age'
    histogram_boxplot(data, "age")

    # Display labeled bar plot for 'tenure'
    labeled_barplot(data, "tenure", perc=True)

    # Display labeled bar plot for 'num_of_products'
    labeled_barplot(data, "num_of_products", perc=True)

    # Display histogram and boxplot for 'balance'
    histogram_boxplot(data, "balance")

    # Display labeled bar plot for 'has_cr_card'
    labeled_barplot(data, "has_cr_card", perc=True)

    # Display histogram and boxplot for 'estimated_salary'
    histogram_boxplot(data, "estimated_salary")

    # Display labeled bar plot for 'is_active_member'
    labeled_barplot(data, "is_active_member", perc=True)

    # Display labeled bar plot for the target variable 'exited'
    labeled_barplot(data, TARGET_COL, perc=True)


def show_salary_barplot_visualization(df: pd.DataFrame) -> None:
    """
    Compares estimated salary to bank balance using a stacked bar plot.

    The 'estimated_salary' and 'balance' columns are grouped into
    intervals of `BALANCE_THRESHOLD` for better visualization.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    barplot_data = df.copy()
    barplot_data["estimated_salary"] = barplot_data["estimated_salary"].apply(
        lambda x: x // BALANCE_THRESHOLD
    )
    barplot_data["balance"] = barplot_data["balance"].apply(
        lambda x: x // BALANCE_THRESHOLD
    )

    # Call stacked barplot with the modified DataFrame
    stacked_barplot(barplot_data, "estimated_salary", "balance")


def show_plot_distributions(df: pd.DataFrame) -> None:
    """
    Generates distribution plots for various features with respect to the
    target variable.

    This function iterates through a predefined list of columns and
    creates comparative distribution plots (histograms and box plots)
    against the target variable. It also includes a stacked bar plot
    for 'num_of_products' vs. the target.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    data_columns: List[str] = [
        "estimated_salary",
        "balance",
        "age",
        "gender",
        "tenure",
        "is_active_member",
        "num_of_products",
        "credit_score",
        "geography",
        "has_cr_card",
    ]

    # Compare column to whether they (customer) exited (the program).
    for col_name in data_columns:
        _distribution_plot_wrt_target(df, col_name, TARGET_COL)

    # Compare Number of Products for customers with a stacked barplot.
    stacked_barplot(df, "num_of_products", TARGET_COL)


def plot_model_performance(
    mod_hist: History, label: str, title: str = ""
) -> None:
    """
    Plots the training and validation loss/accuracy curves from a Keras
    model's history.

    Args:
        mod_hist (History): The history object returned by a Keras model's
                            `fit` method.
        label (str): The metric to plot (e.g., "Loss", "Accuracy").
        title (str, optional): An optional title prefix for the plot.
                               Defaults to "".
    """
    # Standardize label to lowercase to match Keras history keys
    metric_name = label.lower()
    val_metric_name = f"val_{metric_name}"

    fig, ax = plt.subplots()  # Creating a subplot with a figure and axes.

    if metric_name in mod_hist.history:
        ax.plot(mod_hist.history[metric_name], label="Train")
    if val_metric_name in mod_hist.history:
        ax.plot(mod_hist.history[val_metric_name], label="Validation")

    ax.set_title(f"{title.title()} Model: {label.title()}")
    ax.set_ylabel(label.capitalize())
    ax.set_xlabel("Epochs")
    ax.legend(loc="upper right")

    plt.show()


def labeled_barplot(
    data: pd.DataFrame, feature: str, perc: bool = False, fig_size_count: int = 0
) -> None:
    """
    Generates a bar plot for a given feature, optionally displaying
    percentages on top of the bars.

    Args:
        data (pd.DataFrame): The input DataFrame.
        feature (str): The name of the column to plot.
        perc (bool, optional): If True, display percentages instead of
                               counts. Defaults to False.
        fig_size_count (int, optional): Controls the figure size based
                                        on the number of unique categories.
                                        If 0, it's determined by the number
                                        of unique values in the feature.
                                        Defaults to 0.
    """
    title = f"Labeled Bar Plot: {feature.title()}"
    total = len(data[feature])  # length of the column

    if fig_size_count == 0:
        fig_size_count = data[feature].nunique()

    plt.figure(num=title, figsize=(fig_size_count + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature]
        .value_counts()
        .index[:fig_size_count]
        .sort_values(),
    )

    for p in ax.patches:
        if perc:
            label = f"{100 * p.get_height() / total:.1f}%"
        else:
            label = int(p.get_height())
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


def histogram_boxplot(
    data: pd.DataFrame,
    feature: str,
    figsize: Tuple[int, int] = (12, 7),
    kde: bool = False,
    bins: Optional[int] = None,
) -> None:
    """
    Generates a combined histogram and box plot for a numerical feature.

    Args:
        data (pd.DataFrame): The input DataFrame.
        feature (str): The name of the numerical column to plot.
        figsize (Tuple[int, int], optional): Size of the figure.
                                             Defaults to (12, 7).
        kde (bool, optional): If True, plot a kernel density estimate.
                              Defaults to False.
        bins (Optional[int], optional): Number of bins for the histogram.
                                        Defaults to None (auto-determined).
    """
    _, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid = 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,  # creating
    )

    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )

    # boxplot will be created and a star will indicate the mean value
    if bins:
        sns.histplot(
            data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
        )
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)

    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")

    # Add mean to the histogram
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")

    # Add median to the histogram
    plt.show()


def stacked_barplot(data: pd.DataFrame, predictor: str, target: str) -> None:
    """
    Prints category counts and plots a stacked bar chart for two features.

    Args:
        data (pd.DataFrame): The input DataFrame.
        predictor (str): The name of the independent variable column.
        target (str): The name of the target variable column.
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )

    print("-" * PEP8_LINE_LEN)
    print(tab1)
    print("-" * PEP8_LINE_LEN)

    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )

    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def _distribution_plot_wrt_target(
    data: pd.DataFrame, predictor: str, target: str
) -> None:
    """
    Generates a 2x2 grid of plots showing the distribution of a predictor
    variable with respect to the target variable.

    Includes histograms for each target class and box plots (with and
    without outliers) against the target.

    Args:
        data (pd.DataFrame): The input DataFrame.
        predictor (str): The name of the predictor column.
        target (str): The name of the target column.
    """
    _, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title(f"Distribution of {predictor} for {target}={target_uniq[0]}")
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title(f"Distribution of {predictor} for {target}={target_uniq[1]}")
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


def show_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Displays a correlation matrix heatmap for the numerical columns in
    the DataFrame.

    Columns are renamed for better readability in the plot.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    corr_data = df.copy()

    # Efficiently rename columns for better readability
    corr_data.columns = [col.replace("_", " ").title() for col in corr_data.columns]
    
    title = "Correlation Matrix of Bank Customer Churn"

    plt.figure(num=title, figsize=(15, 7))
    sns.heatmap(
        corr_data.corr(numeric_only=True),
        annot=True,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        cmap="Spectral",
    )
    plt.title(title)
    plt.show()


def model_performance_classification(
    local_model: Sequential,
    predictors: pd.DataFrame,
    target: pd.Series,
    threshold: float = CUSTOMER_CHURN_PROB_THRESHOLD,
) -> pd.DataFrame:
    """
    Computes and returns key classification metrics for a given model.

    Metrics include Accuracy, Precision, Recall, F1-Score, and AUC.

    Args:
        local_model (Sequential): The trained Keras Sequential model.
        predictors (pd.DataFrame): The independent variables (features).
        target (pd.Series): The true target labels.
        threshold (float, optional): The probability threshold for
                                     binary classification. Defaults to
                                     `CUSTOMER_CHURN_PROB_THRESHOLD`.

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics.
    """
    # Checking which probabilities are greater than a threshold
    predictions_proba = local_model.predict(predictors, verbose=0)
    pred = (predictions_proba > threshold).astype(int)

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average="weighted")
    recall = recall_score(target, pred, average="weighted")
    f1 = f1_score(target, pred, average="weighted")
    auc = roc_auc_score(target, predictions_proba)

    return pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
            "AUC": [auc],
        }
    )


def show_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> List[str]:
    """
    Generates and displays a detailed classification report in a banner
    format.

    The report includes precision, recall, f1-score, and support for
    each class, along with overall summary metrics like accuracy,
    macro average F1, and weighted average F1.

    Args:
        y_test (pd.Series): The true target labels.
        y_pred (np.ndarray): The predicted labels from the model.

    Returns:
        List[str]: A list of strings representing the formatted
                   classification report.
    """
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict)

    lines: List[str] = []
    for class_label in ["0", "1"]:
        label_name = (
            "Loyal Customer (0)" if class_label == "0" else "Churned Customer (1)"
        )
        lines.append(f"Class: {label_name}")

        # Force the left labels to occupy exactly 12 characters of space
        lines.append(f"  {'Precision':<12} : {report_df.loc['precision', class_label]:.4f}")
        lines.append(f"  {'Recall':<12} : {report_df.loc['recall', class_label]:.4f}")
        lines.append(f"  {'F1-Score':<12} : {report_df.loc['f1-score', class_label]:.4f}")
        lines.append(f"  {'Support':<12} : {int(report_df.loc['support', class_label])}")
        lines.append("")

    # Parse the global overall summary metrics
    lines.append("Overall Summary:")

    # Force the summary labels to occupy exactly 16 characters of space
    # Accuracy is a scalar column in this DF, not a specific row index.
    lines.append(f"  {'Total Accuracy':<16} : {report_dict['accuracy']:.4f}")
    lines.append(f"  {'Macro F1-Avg':<16} : {report_df.loc['f1-score', 'macro avg']:.4f}")
    lines.append(f"  {'Weighted F1-Avg':<16} : {report_df.loc['f1-score', 'weighted avg']:.4f}")

    show_banner("Classification Report", lines)
    return lines