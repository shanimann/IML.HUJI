from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sub

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, size=1000)
    X = UnivariateGaussian()
    X.fit(samples)
    print(f"Mu:{X.mu_}\nVar:{X.var_}")

    # Question 2 - Empirically showing sample mean is consistent
    dist = []
    X2 = UnivariateGaussian()
    sample_count = np.arange(start=10, stop=1001, step=10)
    for size in sample_count:
        X2.fit(samples[:size])
        dist.append(np.absolute(X2.mu_ - 10))

    go.Figure([go.Scatter(x=sample_count, y=dist, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(5) Estimation of Expectation As Function Of"
                        r" Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$distance$",
                  height=600, width = 600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = X.pdf(samples)
    fig = sub.make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=samples, y=pdfs, mode='markers',
                                marker=dict(color="black"), showlegend=False)]) \
        .update_layout(title_text=r"$\text{Empirical PDF Function}$",
                       xaxis_title="$m\\text{ - Sample size}$",
                       yaxis_title="r$PDF- results$",
                       height=800, width=800)
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    sigma = np.array(
        [1, 0.2, 0, 0.5, 0.2, 2, 0, 0, 0, 0, 1, 0, 0.5, 0, 0, 1]).reshape(4,
                                                                          4)
    mu = np.array([0, 0, 4, 0])
    samples = np.random.multivariate_normal(mu, sigma,
                                            size=1000)
    multi = MultivariateGaussian()
    multi.fit(samples)
    print(f"Mu is: {multi.mu_}\nThe Cov Matrix is\n{multi.cov_}")
    # Question 5 - Likelihood evaluation
    F1 = np.linspace(-10, 10, 200)
    F3 = np.linspace(-10, 10, 200)
    log_likelyhoods = []
    for i in F1:
        temp = []
        for j in F3:
            mu_temp = np.array([i, 0, j, 0])
            temp.append(multi.log_likelihood(mu_temp, sigma, samples))
        log_likelyhoods.append(temp)

    trace = go.Heatmap(x=F3, y=F1, z=log_likelyhoods, type='heatmap',
                       colorscale='Viridis')
    data = [trace]
    fig = go.Figure(data=data, layout=go.Layout(
        title=r"$\text{Heat Map for Logliklihood using Mu [0,F1,0,F3]}$",
        xaxis_title="$\\text F3$", yaxis_title="$\\text F1$", height=900,
        width=900))
    fig.show()
    # Question 6 - Maximum likelihood
    log_likelyhoods = np.array(log_likelyhoods).reshape(200, 200)
    f1_index, f3_index = np.unravel_index(np.array(log_likelyhoods).argmax(),
                                          np.array(log_likelyhoods).shape)
    print(F1[f1_index], F3[f3_index])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
