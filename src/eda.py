"""Module to perform basic exploratory data analysis on the dataset."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Load the carbon budget dataset.

    Args:
        None.

    Returns:
        The carbon budget dataset as a pandas DataFrame.

    """
    data = pd.read_csv("data/carbon_budget.csv")
    return data


def return_average_budget(data):
    """Calculate the average budget imbalance.

    Args:
        data: The carbon budget dataset as a pandas DataFrame.

    Returns:
        The average budget imbalance.

    """
    return data["budget imbalance"].mean()


def visualize_budget(data):
    sns.lmplot(x="Year", y="budget imbalance", data=data)
    plt.title("Budget Imbalance Over Time")
    plt.xlabel("Year")
    plt.ylabel("Budget Imbalance (GtCO2)")
    plt.tight_layout()
    plt.show()
    plt.clf()

