import pandas as pd
import numpy as np

from .vis import _get_new_model_instance


def bf_model_comparison(models_data, y):
    df = []
    for i in range(len(models_data)):
        d = {
                'index': f'M_{i+1}',
            }
        for j in range(len(models_data)):
            if i==j:
                d[f'M_{j+1}'] = 1
            else:
                d[f'M_{j + 1}'] = (
                    models_data[j][0].likelihood(models_data[j][1], y) /
                    models_data[i][0].likelihood(models_data[i][1], y)
                )
        df.append(d)
    return pd.DataFrame.from_records(df).set_index('index')


def elpd_loo(model, x, y):
    log_p_is = []
    for mask in range(x.shape[0]):
        x_i = np.concatenate([
            x[0:mask], x[mask+1:]
        ])
        y_i = np.concatenate([
            y[0:mask], y[mask+1:]
        ])
        model_i = _get_new_model_instance(model)
        model_i.fit(x_i, y_i)
        log_p_is.append(np.log(model_i.predict(x[mask]).pdf(y[mask])))

    return np.sum(log_p_is)
