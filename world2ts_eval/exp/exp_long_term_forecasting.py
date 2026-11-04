from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.conditions import NUMERICAL_ONLY, WITH_CONTEXT
from utils.metrics import metric
from utils.tools import EarlyStopping


class Exp_Long_Term_Forecast(Exp_Basic):
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _forward_batch(self, batch):
        (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            metadata,
            events,
            history_time_range,
            forecast_time_range,
        ) = batch
        if self.args.condition == NUMERICAL_ONLY:
            metadata = [""] * len(metadata)
            events = [""] * len(events)
        elif self.args.condition != WITH_CONTEXT:
            raise ValueError(f"Unsupported evaluation condition: {self.args.condition}")
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        decoder_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
        decoder_input = torch.cat(
            [batch_y[:, : self.args.label_len, :], decoder_zeros], dim=1
        )
        output = self.model(
            batch_x,
            batch_x_mark,
            decoder_input,
            batch_y_mark,
            metadata=metadata,
            events=events,
            history_time_range=history_time_range,
            forecast_time_range=forecast_time_range,
        )
        target = batch_y[:, -self.args.pred_len :, :]
        return output, target

    def _validation_loss(self, loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                output, target = self._forward_batch(batch)
                losses.append(criterion(output, target).item())
        self.model.train()
        return float(np.mean(losses))

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, validation_loader = self._get_data("val")
        if hasattr(self.model, "fit_numeric_experts"):
            summary = self.model.fit_numeric_experts(
                train_loader,
                validation_loader,
            )
            print(json.dumps({"castfsr_expert_training": summary}, indent=2))
            return self.model
        checkpoint_dir = Path(self.args.checkpoints) / setting
        early_stopping = EarlyStopping(self.args.patience)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.args.lr_decay, patience=1
        )
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            started = time.time()
            self.model.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                output, target = self._forward_batch(batch)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            validation_loss = self._validation_loss(validation_loader, criterion)
            scheduler.step(validation_loss)
            print(
                f"Epoch {epoch + 1}/{self.args.train_epochs} "
                f"train={train_loss:.6f} val={validation_loss:.6f} "
                f"time={time.time() - started:.1f}s"
            )
            early_stopping(validation_loss, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                break

        state = torch.load(
            checkpoint_dir / "checkpoint.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        return self.model

    def test(self, setting, load_checkpoint=False):
        test_data, test_loader = self._get_data("test")
        if load_checkpoint and self.model.requires_training:
            checkpoint = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)

        predictions = []
        targets = []
        if hasattr(self.model, "start_progress"):
            self.model.start_progress(len(test_data))
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                output, target = self._forward_batch(batch)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        pre_inverse_scores = metric(predictions, targets)
        if self.args.inverse and test_data.scale:
            shape = predictions.shape
            predictions = test_data.inverse_transform(
                predictions.reshape(-1, 1)
            ).reshape(shape)
            targets = test_data.inverse_transform(targets.reshape(-1, 1)).reshape(shape)

        scores = metric(predictions, targets)
        post_inverse_scores = dict(scores)
        scores.update(
            {
                "pre_inverse_metrics": pre_inverse_scores,
                "post_inverse_metrics": post_inverse_scores,
                "model": self.args.model,
                "condition": self.args.condition,
                "series_id": self.args.series_id,
                "num_samples": int(predictions.shape[0]),
                "seq_len": self.args.seq_len,
                "pred_len": self.args.pred_len,
                "stride": test_data.stride,
            }
        )
        if hasattr(self.model, "get_api_usage_metrics"):
            scores["api_usage"] = self.model.get_api_usage_metrics()
        result_dir = Path(self.args.results) / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "pred.npy", predictions)
        np.save(result_dir / "true.npy", targets)
        (result_dir / "metrics.json").write_text(
            json.dumps(scores, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(scores, indent=2, ensure_ascii=False))
        return scores
