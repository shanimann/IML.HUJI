import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"], dayfirst=True)
    X = filter_data(df)
    return X


def filter_data(df):
    # check if data values are valid, filter out invalid rows
    df = df.loc[(df[["Temp"]] > -72).all(axis=1)]
    df = df[df["Month"].isin(range(1, 13))]
    df = df[df["Year"].isin(range(1900, 2022))]
    df['DayOfYear'] = df["Date"].dt.dayofyear
    return df


def filter_by_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    # filter a Dataframe by given country name.
    df = df.loc[(df["Country"] == country)]
    return df.reset_index()


def plot_temp_by_year(X: pd.DataFrame) -> None:
    go.Figure([go.Scatter(x=X['DayOfYear'], y=X['Temp'], mode='markers',
                          name=r'$temp$', marker_color=X['Year'])],
              layout=go.Layout(
                  title=r"$\text{Temp as a function of DayOfYear in Israel}$",
                  xaxis_title="$m\\text{ - Day of Year}$",
                  yaxis_title="r$Temperature$",
                  height=800, width=800)).show()


def plot_std_by_month(X) -> None:
    X = X.groupby('Month')['Temp'].agg(np.std).to_frame()
    px.bar(X, title="Standard Deviation per Month", width=600,
           height=600, labels={'value': 'Standard Deviation'}).show()


def plot_average_by_month_per_country(X):
    X = X.filter(['Country', 'Month', 'Temp'])
    grouped = X.groupby(['Country', 'Month']).agg(X_mean=('Temp', np.mean),
                                                  X_std=('Temp', np.std))

    fig = px.line(data_frame=grouped, x=grouped.index.to_frame()['Month'],
                  y=grouped['X_mean'],
                  color=grouped.index.to_frame()['Country'],
                  error_y=grouped['X_std'])
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Mean"
    )
    fig.show()


def loss_per_k(df):
    y = df['Temp']
    del df['Temp']
    X_train, y_train, X_test, y_test = split_train_test(df, y, 0.75)
    losses = []
    X_train = X_train.to_numpy().flatten()
    X_test = X_test.to_numpy().flatten()
    degrees = [k for k in range(1, 11)]
    for k in degrees:
        model = PolynomialFitting(k)
        model._fit(X_train, y_train)
        loss = model._loss(X_test, y_test)
        losses.append(round(loss, 2))
        print(k, round(loss, 2))
    go.Figure([go.Bar(x=degrees, y=losses, name=r'$Loss$')],
              layout=go.Layout(
                  title=r"$\text{Loss as a function of degree}$",
                  xaxis_title="$m\\text{ k - degree}$",
                  yaxis_title="r$Loss$",
                  height=600, width=600)).show()


def plot_all_countries(df, israel_data, k):
    israel_model = PolynomialFitting(k)
    israel_y = israel_data.filter(['Temp'])
    del israel_data['Temp']
    israel_data = israel_data.to_numpy().flatten()
    israel_model._fit(israel_data, israel_y)
    south_africa_data = filter_by_country(df, 'South Africa')
    sa_y = south_africa_data.filter(['Temp'])
    south_africa_data = south_africa_data.drop(
        columns=['Temp', 'Country', 'index'],
        axis=1)
    sa_loss = israel_model._loss(south_africa_data.to_numpy().flatten(),
                                 sa_y.to_numpy())
    jordan_data = filter_by_country(df, 'Jordan')
    jordan_y = jordan_data.filter(['Temp'])
    jordan_data = jordan_data.drop(columns=['Temp', 'Country', 'index'],
                                   axis=1)
    jordan_loss = israel_model._loss(jordan_data.to_numpy().flatten(),
                                     jordan_y.to_numpy())
    netherlands_data = filter_by_country(df, 'The Netherlands')
    netherlands_y = netherlands_data.filter(['Temp'])
    netherlands_data = netherlands_data.drop(
        columns=['Temp', 'Country', 'index'],
        axis=1)
    netherlands_loss = israel_model._loss(
        netherlands_data.to_numpy().flatten(), netherlands_y.to_numpy())
    losses = {'South Africa': sa_loss, 'Jordan': jordan_loss,
              'The Netherlands': netherlands_loss}
    x_axis = list(losses.keys())
    y_axis = list(losses.values())
    go.Figure([go.Bar(x=x_axis, y=y_axis, name=r'$Loss$')],
              layout=go.Layout(
                  title=r"$\text{Loss per country with degree 5}$",
                  xaxis_title="$m\\text{ Country}$",
                  yaxis_title="r$Loss$",
                  height=600, width=600)).show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data(
        r"C:\Users\shani\PycharmProjects\IML.HUJI\datasets\City_Temperature.csv")
    X.to_csv(r'C:\Users\shani\OneDrive\Desktop\filtered_city_data.csv')

    # Question 2 - Exploring data for specific country
    israel_data = filter_by_country(X, 'Israel')
    israel_data.to_csv(r'C:\Users\shani\OneDrive\Desktop\filtered_israel.csv')
    plot_temp_by_year(israel_data)
    plot_std_by_month(israel_data)
    # #Question 3 - Exploring differences between countries
    plot_average_by_month_per_country(X)

    # Question 4 - Fitting model for different values of `k`
    loss_per_k(israel_data.filter(['DayOfYear', 'Temp']))

    # Question 5 - Evaluating fitted model on different countries
    plot_all_countries(X.filter(['DayOfYear', 'Temp', 'Country']),
                       israel_data.filter(['DayOfYear', 'Temp']), 5)
