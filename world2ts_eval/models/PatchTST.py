import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    def __init__(self, input_size, target_window, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(input_size, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """Forecasting-only PatchTST adapted from THUML Time-Series-Library."""

    requires_training = True

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        patch_len = configs.patch_len
        patch_stride = configs.patch_stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            patch_len,
            patch_stride,
            patch_stride,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )
        patch_count = int((configs.seq_len - patch_len) / patch_stride + 2)
        self.head = FlattenHead(
            configs.d_model * patch_count,
            configs.pred_len,
            configs.dropout,
        )

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        normalized = x_enc - means
        stdev = torch.sqrt(
            torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = normalized / stdev

        encoded, n_vars = self.patch_embedding(normalized.permute(0, 2, 1))
        encoded, _ = self.encoder(encoded)
        encoded = torch.reshape(
            encoded, (-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        )
        encoded = encoded.permute(0, 1, 3, 2)
        output = self.head(encoded).permute(0, 2, 1)
        return output * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

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
        return self.forecast(x_enc)[:, -self.pred_len :, :]
