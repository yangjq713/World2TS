from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


@contextmanager
def _open_arrow(path: Path):
    try:
        with pa.memory_map(str(path), "r") as source:
            with ipc.open_file(source) as reader:
                yield reader
    except pa.ArrowInvalid:
        with pa.memory_map(str(path), "r") as source:
            with ipc.open_stream(source) as reader:
                yield reader


def _batches(reader):
    if hasattr(reader, "num_record_batches"):
        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)
        return
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


def _series_candidates(item_id: str, target):
    if not target or not isinstance(target[0], list):
        yield item_id, target
        return
    for channel_index, channel in enumerate(target):
        yield f"{item_id}__target_{channel_index}", channel


def _load_series(path: Path, series_id: str):
    with _open_arrow(path) as reader:
        for batch in _batches(reader):
            names = batch.schema.names
            columns = [batch.column(i) for i in range(batch.num_columns)]
            for row_index in range(batch.num_rows):
                row = {
                    name: columns[column_index][row_index].as_py()
                    for column_index, name in enumerate(names)
                }
                item_id = str(row["item_id"])
                for candidate_id, target in _series_candidates(item_id, row["target"]):
                    if candidate_id == series_id:
                        return (
                            np.asarray(target, dtype=np.float32),
                            pd.Timestamp(row["start"]),
                            str(row["freq"]),
                            row.get("timestamps"),
                        )
    raise ValueError(f"Series {series_id!r} was not found in {path}")


def _interpolate_nan_values(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    series_id: str,
) -> tuple[np.ndarray, int]:
    """Linearly interpolate NaNs and extend boundary gaps with nearest values."""
    if values.ndim != 1:
        raise ValueError(f"Series {series_id!r} target must be one-dimensional")
    if len(values) != len(dates):
        raise ValueError(
            f"Series {series_id!r} has {len(values)} values but {len(dates)} timestamps"
        )
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError(
            f"Series {series_id!r} timestamps must be unique and strictly increasing"
        )
    if np.isinf(values).any():
        raise ValueError(f"Series {series_id!r} contains infinite target values")

    nan_mask = np.isnan(values)
    num_interpolated = int(nan_mask.sum())
    if num_interpolated == 0:
        return values, 0
    if num_interpolated == len(values):
        raise ValueError(f"Series {series_id!r} target contains only NaN values")

    series = pd.Series(values.astype(np.float64, copy=False), index=dates)
    interpolated = series.interpolate(method="time", limit_area="inside").ffill().bfill()
    result = interpolated.to_numpy(dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Series {series_id!r} target interpolation was incomplete")
    return result, num_interpolated


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _pandas_frequency(freq: str) -> str:
    aliases = {
        "1d": "D",
        "daily": "D",
        "1w": "W",
        "weekly": "W",
        "1h": "h",
        "hourly": "h",
    }
    return aliases.get(freq.strip().casefold(), freq)


def _regular_dates(start: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    start = _normalize_timestamp(start)
    normalized = freq.strip().casefold()
    fixed_deltas = {
        "15min": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "h": pd.Timedelta(hours=1),
        "hourly": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "d": pd.Timedelta(days=1),
        "daily": pd.Timedelta(days=1),
        "1w": pd.Timedelta(weeks=1),
        "w": pd.Timedelta(weeks=1),
        "weekly": pd.Timedelta(weeks=1),
    }
    if normalized in fixed_deltas:
        delta = fixed_deltas[normalized]
        return pd.DatetimeIndex([start + index * delta for index in range(periods)])
    offset = pd.tseries.frequencies.to_offset(_pandas_frequency(freq))
    return pd.date_range(start=start, periods=periods, freq=offset)


class Dataset_Custom(Dataset):
    """TSLib-style univariate forecasting dataset backed by World2TS artifacts."""

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        data_path="data.arrow",
        scale=True,
        timeenc=0,
        freq="auto",
        stride=None,
        series_id=None,
        top_k_events=10,
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split: {flag}")
        self.args = args
        self.seq_len, self.label_len, self.pred_len = size or (96, 48, 12)
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.root_path = Path(root_path)
        self.data_path = self.root_path / data_path
        self.scale = scale
        self.timeenc = timeenc
        self.requested_freq = freq
        self.requested_stride = stride
        self.series_id = series_id
        self.top_k_events = top_k_events
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        values, start, arrow_freq, timestamps = _load_series(
            self.data_path, self.series_id
        )
        self.freq = arrow_freq if self.requested_freq == "auto" else self.requested_freq
        self.stride = self.requested_stride or (1 if self.flag == "train" else 24)
        self.dates = (
            pd.DatetimeIndex([_normalize_timestamp(value) for value in timestamps])
            if timestamps is not None
            else _regular_dates(start, len(values), self.freq)
        )
        values, self.num_interpolated_values = _interpolate_nan_values(
            values,
            self.dates,
            self.series_id,
        )

        num_train = int(len(values) * 0.7)
        num_test = int(len(values) * 0.2)
        num_val = len(values) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(values) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(values)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        values = values.reshape(-1, 1)
        if self.scale:
            self.scaler.fit(values[border1s[0] : border2s[0]])
            values = self.scaler.transform(values)

        self.data_x = values[border1:border2]
        self.data_y = values[border1:border2]
        self.data_dates = self.dates[border1:border2]
        self.data_stamp = time_features(self.data_dates, timeenc=self.timeenc)
        self.metadata_text, self.events = self._load_context()
        self.sample_starts = self._build_sample_starts()
        if self.flag == "test" and not self.sample_starts:
            raise ValueError(
                f"Series {self.series_id!r} has no test windows containing a "
                "linked event inside the numeric history window"
            )

    def _load_context(self):
        dataset_info_path = self.root_path / "dataset_info.json"
        dataset_info = (
            json.loads(dataset_info_path.read_text(encoding="utf-8"))
            if dataset_info_path.exists()
            else {}
        )
        series_rows = {
            row["series_id"]: row
            for row in _read_jsonl(self.root_path / "unified_event_series.jsonl")
        }
        series_row = series_rows.get(self.series_id, {})
        metadata_sections = []
        if dataset_info.get("description"):
            metadata_sections.append(
                "Dataset description:\n" + dataset_info["description"]
            )
        if series_row.get("description"):
            metadata_sections.append(
                "Series description:\n" + series_row["description"]
            )
        metadata_text = "\n\n".join(metadata_sections)

        linked_ids = {link["event_id"] for link in series_row.get("event_links", [])}
        events = []
        for event in _read_jsonl(self.root_path / "unified_events.jsonl"):
            if event["event_id"] in linked_ids:
                events.append(
                    {
                        "event_id": event["event_id"],
                        "timestamp": _normalize_timestamp(event["timestamp"]),
                        "summary": event["summary"],
                    }
                )
        events.sort(key=lambda event: event["timestamp"])
        return metadata_text, events

    def _event_text(
        self,
        history_start: pd.Timestamp,
        forecast_start: pd.Timestamp,
    ) -> str:
        eligible = [
            event
            for event in self.events
            if history_start <= event["timestamp"] < forecast_start
        ][-self.top_k_events :]
        return "\n".join(
            f"- [{event['timestamp'].isoformat()}] {event['summary']}"
            for event in eligible
        )

    def _build_sample_starts(self) -> list[int]:
        available = len(self.data_x) - self.seq_len - self.pred_len
        if available < 0:
            return []
        starts = list(range(0, available + 1, self.stride))
        if self.flag != "test":
            return starts
        return [
            start
            for start in starts
            if self._event_text(
                self.data_dates[start],
                self.data_dates[start + self.seq_len],
            )
        ]

    def __getitem__(self, index):
        s_begin = self.sample_starts[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        forecast_start = self.data_dates[s_end]
        history_time_range = json.dumps(
            {
                "start": self.data_dates[s_begin].isoformat(),
                "end": self.data_dates[s_end - 1].isoformat(),
            }
        )
        forecast_time_range = json.dumps(
            {
                "start": self.data_dates[s_end].isoformat(),
                "end": self.data_dates[r_end - 1].isoformat(),
            }
        )
        return (
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
            self.metadata_text,
            self._event_text(self.data_dates[s_begin], forecast_start),
            history_time_range,
            forecast_time_range,
        )

    def __len__(self):
        return len(self.sample_starts)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
