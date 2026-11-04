import torch
from torch import nn


class Model(nn.Module):
    requires_training = False

    def __init__(self, configs):
        super().__init__()
        self.model_name = configs.model
        self.pred_len = configs.pred_len
        self.seasonal_period = configs.seasonal_period

    def forecast(self, x_enc):
        if self.model_name == "LastValue":
            return x_enc[:, -1:, :].repeat(1, self.pred_len, 1)
        if self.seasonal_period > x_enc.shape[1]:
            raise ValueError("seasonal_period cannot exceed seq_len")
        season = x_enc[:, -self.seasonal_period :, :]
        indices = torch.arange(self.pred_len, device=x_enc.device)
        indices = indices.remainder(self.seasonal_period)
        return season[:, indices, :]

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        metadata=None,
        events=None,
        history_time_range=None,
        forecast_time_range=None,
    ):
        return self.forecast(x_enc)
