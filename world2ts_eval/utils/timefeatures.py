import numpy as np
import pandas as pd


def time_features(dates, timeenc=0):
    index = pd.DatetimeIndex(dates)
    raw = np.column_stack([index.month, index.day, index.dayofweek, index.hour]).astype(
        np.float32
    )
    if timeenc == 0:
        return raw
    scales = np.asarray([12.0, 31.0, 7.0, 24.0], dtype=np.float32)
    return raw / scales - 0.5
