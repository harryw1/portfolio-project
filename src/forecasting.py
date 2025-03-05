"""Module for forecasting future carbon budget components."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_future_years(start_year, end_year):
    """Generate a list of future years from start_year to end_year.

    Args:
        start_year (int): The first year to include in the forecast range.
        end_year (int): The last year to include in the forecast range.

    Returns:
        numpy.ndarray: An array of years from start_year to end_year.

    """
    years = np.arange(start_year, end_year + 1).reshape(-1, 1)
    return years


def forecast_component(model, future_years):
    """Forecast future values for a single component.

    Args:
        model (sklearn.linear_model): A trained linear regression model.
        future_years (numpy.ndarray): 2D array of future years to forecast.

    Returns:
        numpy.ndarray: Forecasted values for the component.

    """
    return model.predict(future_years)


def forecast_all(models, future_years):
    """Forecast future values for all components.

    Args:
        models (dict): A dictionary of trained linear regression models.
        future_years (numpy.ndarray): 2D array of future years to forecast.

    Returns:
        pandas.DataFrame: Forecasted values for all components.

    """
    forecast_df = pd.DataFrame({"Year": future_years.flatten()})

    for component, model in models.items():
        forecast_df[component] = forecast_component(model, future_years)

    return forecast_df


def calculate_budget_imbalance(forecast_df):
    """Calculate the budget imbalance for each year.

    Args:
        forecast_df (pandas.DataFrame): Forecasted values for all components.

    Returns:
        pandas.DataFrame: The input dataframe with an additional "Budget Imbalance" column.

    """
    result_df = forecast_df.copy()

    # Sources (positive terms in the budget)
    sources = ["fossil emissions excluding carbonation", "land-use change emissions"]

    # Sinks (negative terms in the budget)
    sinks = ["atmospheric growth", "ocean sink", "land sink", "cement carbonation sink"]

    result_df["budget imbalance"] = result_df[sources].sum(axis=1) - result_df[
        sinks
    ].sum(axis=1)

    return result_df


def plot_forecast(historical, forecast, components=None):
    """Plot historical data and forecasted values.

    Args:
        historical (pandas.DataFrame): Historical data.
        forecast (pandas.DataFrame): Forecasted values.
        components (list): Components to plot. If None, plot all components.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.

    """
    if components is None:
        # Exclude 'Year' and 'budget imbalance'
        components = [
            col for col in forecast.columns if col not in ["Year", "budget imbalance"]
        ]

    fig, axes = plt.subplots(len(components), 1, figsize=(10, 3 * len(components)))

    if len(components) == 1:
        axes = [axes]  # Make axes iterable when there's only one component

    for ax, component in zip(axes, components):
        # Plot historical data
        ax.plot(historical["Year"], historical[component], "o-", label="Historical")

        # Plot forecast
        ax.plot(forecast["Year"], forecast[component], "--", label="Forecast")

        ax.set_title(component)
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    return fig


def pipeline(historical, models, start_year, end_year):
    """Generate forecasts for all components and calculate budget imbalance.

    Args:
        historical (pandas.DataFrame): Historical data.
        models (dict): A dictionary of trained linear regression models.
        start_year (int): The first year to include in the forecast range.
        end_year (int): The last year to include in the forecast range.

    Returns:
        tuple (forecast, fig): Forecasted values for all components and
            the figure object for the plot.

    """
    future_years = generate_future_years(start_year, end_year)
    forecast = forecast_all(models, future_years)
    forecast = calculate_budget_imbalance(forecast)
    fig = plot_forecast(historical, forecast)
    return forecast, fig
