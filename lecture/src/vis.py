from matplotlib import pyplot as plt
import numpy as np


def plot_model(model, x, y, interval=0.95):
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
        x[:, 0],
        lower_interval,
        upper_interval,
        color='cornflowerblue',
        alpha=0.5,
        label=f"Credible Interval"
    )

    plt.plot(x[:, 0], predictive_mean, color="red", label="Mean output")
    plt.text(1, 0, f'Percent of observations within CI: {ci_ratio * 100:.2f}%',
             fontsize=15, horizontalalignment='right',
             verticalalignment='bottom', transform=plt.gca().transAxes)

    plt.scatter(
                x[:, 0], y,
                marker='+', s=100,
                alpha=1,
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc='upper right')

    plt.show()