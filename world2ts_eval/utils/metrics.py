import numpy as np


def metric(pred, true):
    error = pred - true
    absolute_error = np.abs(error)
    squared_error = error**2
    denominator = np.maximum(np.abs(true), 1e-8)
    smape_denominator = np.maximum(np.abs(pred) + np.abs(true), 1e-8)
    return {
        "mae": float(np.mean(absolute_error)),
        "mse": float(np.mean(squared_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mape": float(np.mean(absolute_error / denominator)),
        "smape": float(np.mean(2.0 * absolute_error / smape_denominator)),
    }
