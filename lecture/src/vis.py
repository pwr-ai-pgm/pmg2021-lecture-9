import math

from matplotlib import pyplot as plt
import numpy as np
from arviz import kde, plot_kde

from .blr import ConjugateBayesLinReg, FullConjugateBayesLinReg
from .data import _data_fun


def plot_model(model, x, x_org, y, interval=0.95, data_fun=_data_fun):
    fig = plt.figure(figsize=(12, 8))

    predictive = model.predict(x)
    predictive_mean = predictive.mean()
    lower_interval, upper_interval = predictive.interval(interval)
    ci_ratio = np.sum(
        (lower_interval <= y) &
        (y <= upper_interval)
    ) / len(y)

    plt.ylim(
        min(np.min(lower_interval), np.min(y)) - 0.5,
        max(np.max(upper_interval), np.max(y)) + 0.5,
    )

    plt.fill_between(
        x_org,
        lower_interval,
        upper_interval,
        color='cornflowerblue',
        alpha=0.5,
        label=f"Credible Interval {interval}"
    )

    predictive_std = predictive.std()
    plt.fill_between(
        x_org,
        predictive_mean - predictive_std,
        predictive_mean + predictive_std,
        color='green',
        alpha=0.5,
        label=f"+/- Standard deviation"
    )

    plt.plot(x_org, predictive_mean, color="red", label="Mean output")
    plt.text(1, 0, f'Percent of observations within CI: {ci_ratio * 100:.2f}%',
             fontsize=15, horizontalalignment='right',
             verticalalignment='bottom', transform=plt.gca().transAxes)

    plt.scatter(
                x_org, y,
                marker='+', s=100,
                alpha=1,
            )
    xlim = plt.gca().get_xlim()
    x = np.linspace(xlim[0], xlim[1], 1000)
    plt.plot(x, np.vectorize(data_fun)(x), c='k', linestyle='--', alpha=0.75,
             label='Mean of data generating process')
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc='upper right')

    plt.show()


def plot_ppc(predictive, y, S=1000, title=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    y_sampled = predictive.rvs(size=(S, y.shape[0]), random_state=1)
    linewidth = 4
    plot_kde(
        y.flatten(),
        label="Observed",
        plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
        fill_kwargs={"alpha": 0},
        ax=ax,
    )
    pp_densities = []
    pp_xs = []
    for vals in y_sampled:
        vals = np.array([vals]).flatten()
        pp_x, pp_density = kde(vals)
        pp_densities.append(pp_density)
        pp_xs.append(pp_x)

    ax.plot(
        np.transpose(pp_xs),
        np.transpose(pp_densities),
        **{"color": 'b', "alpha": 0.1, "linewidth": 0.15 * linewidth},
    )
    ax.plot([], color='b', label="Posterior predictive")

    plt.xlabel('y')
    plt.xlabel('density')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def _get_p_value(dist, observed):
    p_l = np.sum(dist <= observed) / len(dist)
    p_r = np.sum(dist >= observed) / len(dist)
    return 2 * min(p_l, p_r)


def plot_posterior_test_statistics(predictive, y, test_stat, S=1000, title=None):
    n_test = len(test_stat)
    fig, axs = plt.subplots(
        ncols=math.ceil(math.sqrt(n_test)),
        nrows=math.ceil(math.sqrt(n_test)),
        figsize=(12, 8)
    )

    axs = axs.ravel()
    i = 0
    y_sampled = predictive.rvs(size=(S, y.shape[0]), random_state=1)
    for stat_name, stat_fun in test_stat.items():
        stat_vals = []
        for vals in y_sampled:
            stat_vals.append(stat_fun(vals))
        axs[i].hist(stat_vals, bins=20, alpha=0.5,
                    label='Posterior test statistics distribution')
        axs[i].axvline(stat_fun(y), c='k', label='Observed test statistics')
        axs[i].set_title(
            f'{stat_name} - p={_get_p_value(np.array(stat_vals), stat_fun(y))}'
        )
        i += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    if title:
        fig.suptitle(title)
    plt.show()


def _get_new_model_instance(model):
    klass = model.__class__
    if isinstance(model, FullConjugateBayesLinReg):
        return ConjugateBayesLinReg(
            n_features=model.n_features,
            alpha=model.alpha,
            lmbda=model.lmbda
        )
    elif isinstance(model, ConjugateBayesLinReg):
        return klass(
            n_features=model.n_features,
            alpha=model.alpha,
            lmbda=model.lmbda
        )
    else:
        raise ValueError(f'Unknown model class {klass}')


def calculate_cdf_mc(predictive, y, S=1000):
    y_sampled = predictive.rvs(size=(S, 1), random_state=1)
    return np.sum(y_sampled <= y) / S


def plot_pit(model, x, y, S=1000, title=None):
    p_is = []
    for mask in range(x.shape[0]):
        x_i = np.concatenate([
            x[0:mask], x[mask+1:]
        ])
        y_i = np.concatenate([
            y[0:mask], y[mask+1:]
        ])
        model_i = _get_new_model_instance(model)
        model_i.fit(x_i, y_i)
        p_is.append(model_i.predict(x[mask]).cdf(y[mask]))
    linewidth = 4
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_kde(
        np.array(p_is),
        label="PIT density",
        plot_kwargs={"color": "k", "linewidth": linewidth, "zorder": 3},
        fill_kwargs={"alpha": 0},
        ax=ax,
    )

    uni_densities = []
    uni_xs = []
    rng = np.random.default_rng(1)
    for _ in range(S):
        uni_x, uni_density = kde(rng.random(y.shape[0]))
        uni_densities.append(uni_density)
        uni_xs.append(uni_x)

    ax.plot(
        np.transpose(uni_xs),
        np.transpose(uni_densities),
        **{"color": 'b', "alpha": 0.1, "linewidth": 0.15 * linewidth},
    )
    ax.plot([], color='b', label="Uniform empirical densities")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_elpd_loo_comparison(models_data, x_org, y, title=None):
    elpds = {}
    for i in range(len(models_data)):
        log_p_is = []
        for mask in range(y.shape[0]):
            x_i = np.concatenate([
                models_data[i][1][0:mask], models_data[i][1][mask+1:]
            ])
            y_i = np.concatenate([
                y[0:mask], y[mask+1:]
            ])
            model_i = _get_new_model_instance(models_data[i][0])
            model_i.fit(x_i, y_i)
            log_p_is.append(np.log(model_i.predict(models_data[i][1][mask]).pdf(y[mask])))
        elpds[i] = np.array(log_p_is)


    fig, axs = plt.subplots(
        ncols=len(models_data),
        nrows=len(models_data),
        figsize=(16, 12)
    )

    for i in range(len(models_data)):
        for j in range(len(models_data)):
            if i == 0:
                axs[i, j].set_title(f'M_{j+1}')
            if j == 0:
                axs[i, j].set_ylabel(f'M_{i+1}')

            if i == j:
                continue
            axs[i, j].scatter(x_org, elpds[j] - elpds[i], alpha=0.75, s=4)
            axs[i, j].set_xlabel('x')
            axs[i, j].axhline(0, linestyle='--', c='k', alpha=0.5)

    if title:
        plt.title(title)
    plt.show()

