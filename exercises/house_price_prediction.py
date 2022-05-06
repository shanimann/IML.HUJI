from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.utils.utils import split_train_test
from IMLearn.learners.regressors import linear_regression
from IMLearn.metrics.loss_functions import mean_square_error

pio.templates.default = "simple_white"
ZERO_COLUMN_NAMES = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                     'floors', 'grade', 'condition', 'sqft_above',
                     'sqft_basement', 'yr_built', 'zipcode', 'sqft_living15',
                     'sqft_lot15', 'price']
COLUMNS_TO_REMOVE = ['yr_built', 'id', 'date', 'lat', 'long']


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).drop_duplicates()
    df = filter_data(df)
    return df.drop("price", axis=1), df.filter(['price'])


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    # filter out invalid rows by checking invalid values for each feature.
    df = df[df["condition"].isin([1, 2, 3, 4, 5])]
    df = df[df["grade"].isin(range(1, 13))]
    df = df[df["waterfront"].isin([0, 1])]
    # filtering all rows returning negative values in specific features.
    for col in ZERO_COLUMN_NAMES:
        df = df[df[col] >= 0]
    df = pd.get_dummies(df, columns=['zipcode'],
                        prefix='zipcode_')
    df['relative_year_built'] = df['yr_built'] / 2020
    df = df.drop(columns=COLUMNS_TO_REMOVE, axis=1)
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = y.std()
    correlations = []
    for (feature, feature_data) in X.iteritems():
        if 'zipcode' not in feature:
            temp_df = feature_data.to_frame()
            corr = float(temp_df[feature].cov(y['price']) / (feature_data.std() * y_std))
            correlations.append((feature, corr))
            print(feature, corr)
    i = 1
    for (feature, feature_data) in X.iteritems():
        if 'zipcode' not in feature:
            go.Figure(
                [go.Scatter(x=feature_data.tolist(), y=y['price'].tolist(),
                            mode='markers')],
                layout=go.Layout(
                    title_text=r"$\text{Pearson Correlation by feature}$",
                    xaxis_title="$m\\feature$",
                    height=600)).write_image(output_path + f"\{feature}.png")
            i += 1


def fit_over_percentages(X, y):
    percentages = [i for i in range(10, 101)]
    X_train, y_train, X_test, y_test = split_train_test(X, y, 0.75)
    lr = LinearRegression()
    means = []
    variances = []
    for p in percentages:
        p_losses = []
        for i in range(10):
            X_p_train, y_p_train, X_p_test, y_p_test = split_train_test(
                X_train, y_train,
                p / 100)
            lr._fit(X_p_train, y_p_train)
            y_hats = lr._predict(X_test)
            p_losses.append(mean_square_error(y_test, y_hats))
        means.append(np.mean(np.array(p_losses)))
        variances.append(np.var(np.array(p_losses)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentages,
                             y=means,
                             mode='lines+markers',
                             name='mean loss'
                             ))

    fig.update_layout(
        title='Average Loss as a function of %p of training size',
        xaxis=dict(
            title='p - % of training set',
            titlefont_size=16,
            tickfont_size=14),
        yaxis=dict(
            title='Average loss',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ))
    fig.add_traces(
        [go.Scatter(x=percentages, y=means + 2 * np.sqrt(np.array(variances)),
                    mode='lines', line_color='rgba(0,0,0,0)',
                    name='Confidence interval',
                    showlegend=False),
         go.Scatter(x=percentages, y=means - 2 * np.sqrt(np.array(variances)),
                    mode='lines', line_color='rgba(0,0,0,0)',
                    name='Confidence interval',
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)')])

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(
        r"C:\Users\shani\PycharmProjects\IML.HUJI\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, r"C:\Users\shani\OneDrive\Desktop\ex1 IML")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_over_percentages(X, y)
