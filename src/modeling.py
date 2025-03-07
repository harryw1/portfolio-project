"""Module to model the dataset and fit a linear regression model."""

import numpy as np
from sklearn.linear_model import LinearRegression


def get_features(data, year="Year", exclude=None):
    """Extract the names of the components to model.

    Args:
        data: The carbon budget dataset as a pandas DataFrame.
        year: The name of the year column. (default: "Year")
        exclude: The name of the column to exclude from the features. (default: None)

    Returns:
        features: The names of the components to model.

    """
    if exclude is None:
        exclude = []

    if year not in exclude:
        exclude.append(year)

    features = [col for col in data.columns if col not in exclude]

    return features


def train_component(data, component, year="Year"):
    """Create features and train a model for a single component.

    Args:
        data: The carbon budget dataset as a pandas DataFrame.
        component: The names of the components to model.
        year: The name of the year column. (default: "Year")

    Returns:
        model: The fitted linear regression model.
        score: The R^2 score of the model.

    """
    X = np.array(data[year]).reshape(-1, 1)
    y = np.array(data[component])
    model = LinearRegression().fit(X, y)
    score = model.score(X, y)
    return model, score


def train_all(data, year="Year", exclude=None):
    """Create features and train a model for all components.

    Args:
        data: The carbon budget dataset as a pandas DataFrame.
        year: The name of the year column. (default: "Year")
        exclude: The name of the column to exclude from the features. (default: None)

    Returns:
        models: A dictionary of fitted linear regression models.
        scores: A dictionary of R^2 scores for each model.

    """
    if exclude is None:
        exclude = []

    features = get_features(data, year, exclude)
    models = {}
    scores = {}

    for component in features:
        model, score = train_component(data, component, year)
        models[component] = model
        scores[component] = score

    return models, scores