import os
import gc
import json
import torch
from glob import glob
from typing import Optional, List
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from itertools import combinations
from sklearn import preprocessing
import joblib
from common_utils_dev import make_dirs, to_parquet, to_abs_path, get_filename_by_path
from pandarallel import pandarallel
from dataclasses import dataclass


CONFIG = {
    "rawdata_dir": to_abs_path(__file__, "../../storage/dataset/rawdata/cleaned/"),
    "data_store_dir": to_abs_path(__file__, "../../storage/dataset/dataset/v001/"),
    "lookahead_window": 30,
    "train_ratio": 0.80,
    "scaler_type": "StandardScaler",
    "winsorize_threshold": 6,
    "query_min_start_dt": "2018-06-01",
}
OHLC = ["open", "high", "low", "close"]


@dataclass
class DatasetBuilder:
    # Defined in running code.
    # Need to give below parameters when build in trader
    tradable_coins: Optional[List] = None
    features_columns: Optional[List] = None
    feature_scaler: Optional[preprocessing.StandardScaler] = None
    label_scaler: Optional[preprocessing.StandardScaler] = None

    def build_rawdata(self, file_names, query_min_start_dt):
        def _load_rawdata_row(file_name):
            rawdata = pd.read_parquet(file_name)
            rawdata.index = pd.to_datetime(rawdata.index)
            rawdata = rawdata[query_min_start_dt:]

            return rawdata

        rawdata = {}
        for file_name in tqdm(file_names):
            coin = get_filename_by_path(file_name)
            rawdata[coin] = _load_rawdata_row(file_name=file_name)

        rawdata = pd.concat(rawdata, axis=1).sort_index()

        self.tradable_coins = sorted(rawdata.columns.levels[0].tolist())

        return rawdata[self.tradable_coins]

    def _build_feature_by_rawdata_row(self, rawdata_row, scaler_target=True):
        if scaler_target is True:
            returns_1320m = (
                rawdata_row[OHLC]
                .pct_change(1320, fill_method=None)
                .rename(columns={key: key + "_return(1320)" for key in OHLC})
            ).dropna()

            returns_600m = (
                (
                    rawdata_row[OHLC]
                    .pct_change(600, fill_method=None)
                    .rename(columns={key: key + "_return(600)" for key in OHLC})
                )
                .dropna()
                .reindex(returns_1320m.index)
            )

            returns_240m = (
                (
                    rawdata_row[OHLC]
                    .pct_change(240, fill_method=None)
                    .rename(columns={key: key + "_return(240)" for key in OHLC})
                )
                .dropna()
                .reindex(returns_1320m.index)
            )

            returns_120m = (
                (
                    rawdata_row[OHLC]
                    .pct_change(120, fill_method=None)
                    .rename(columns={key: key + "_return(120)" for key in OHLC})
                )
                .dropna()
                .reindex(returns_1320m.index)
            )

            returns_1m = (
                (
                    rawdata_row[OHLC]
                    .pct_change(1, fill_method=None)
                    .rename(columns={key: key + "_return(1)" for key in OHLC})
                )
                .dropna()
                .reindex(returns_1320m.index)
            )

            mean_volume_changes_120m = (
                (rawdata_row["volume"] + 1e-7)
                .rolling(120)
                .mean()
                .pct_change(1, fill_method=None)
                .dropna()
                .reindex(returns_1320m.index)
                .rename("mean_volume_changes_120m")
            ).clip(-10, 10)

            volume_changes_1m = (
                (np.log(rawdata_row["volume"] + 1) + 1e-7)
                .pct_change(1, fill_method=None)
                .dropna()
                .reindex(returns_1320m.index)
                .rename("volume_changes_1m")
            ).clip(-10, 10)

            inner_changes = []
            for column_pair in sorted(list(combinations(OHLC, 2))):
                inner_changes.append(
                    rawdata_row[list(column_pair)]
                    .pct_change(1, axis=1, fill_method=None)[column_pair[-1]]
                    .rename("_".join(column_pair) + "_change")
                )

            inner_changes = pd.concat(inner_changes, axis=1)

            inner_changes_shift_120m = (
                inner_changes.shift(120)
                .dropna()
                .reindex(returns_1320m.index)
                .rename(
                    columns={
                        column: column + "_120m" for column in inner_changes.columns
                    }
                )
            )

            inner_changes = inner_changes.dropna().reindex(returns_1320m.index)

            return (
                pd.concat(
                    [
                        returns_1320m,
                        returns_600m,
                        returns_240m,
                        returns_120m,
                        returns_1m,
                        inner_changes,
                        inner_changes_shift_120m,
                        mean_volume_changes_120m,
                        volume_changes_1m,
                    ],
                    axis=1,
                )
                .sort_index()
                .dropna()
            )

        else:
            volume_exists = (
                ((rawdata_row["volume"] == 0) * 1.0)
                .rename("volume_exists")
                .to_frame()
                .sort_index()
            )

            hour_to_8class = {idx: idx // 3 for idx in range(24)}
            hours = pd.DataFrame(
                torch.nn.functional.one_hot(
                    torch.tensor(
                        rawdata_row.index.hour.map(lambda x: hour_to_8class[x])
                    ),
                    num_classes=8,
                )
                .float()
                .numpy(),
                index=rawdata_row.index,
            ).rename(columns={idx: f"8class_{idx}" for idx in range(8)})

            return pd.concat([volume_exists, hours,], axis=1).sort_index().dropna()

    def build_features(self, rawdata):
        features = {}
        class_features = {}
        for coin in tqdm(self.tradable_coins):
            features[coin] = self._build_feature_by_rawdata_row(
                rawdata_row=rawdata[coin], scaler_target=True
            )

            class_features[coin] = self._build_feature_by_rawdata_row(
                rawdata_row=rawdata[coin], scaler_target=False
            )

        features = pd.concat(features, axis=1).sort_index()[self.tradable_coins]
        class_features = pd.concat(class_features, axis=1).sort_index()[
            self.tradable_coins
        ]

        # reindex by common_index
        common_index = features.index & class_features.index
        features = features.reindex(common_index)
        class_features = class_features.reindex(common_index)

        if self.features_columns is None:
            self.features_columns = sorted(
                features.columns.tolist() + class_features.columns.tolist()
            )

        return (
            features[
                [
                    feature
                    for feature in self.features_columns
                    if feature in features.columns
                ]
            ],
            class_features[
                [
                    feature
                    for feature in self.features_columns
                    if feature in class_features.columns
                ]
            ],
        )

    def build_scaler(self, data, scaler_type):
        scaler = getattr(preprocessing, scaler_type)()
        scaler.fit(data)

        return scaler

    def preprocess_features(self, features, winsorize_threshold):
        assert self.feature_scaler is not None

        features = pd.DataFrame(
            self.feature_scaler.transform(features),
            index=features.index,
            columns=features.columns,
        )

        if winsorize_threshold is not None:
            features = (
                features.clip(-winsorize_threshold, winsorize_threshold)
                / winsorize_threshold
            )

        return features

    def preprocess_labels(self, labels, winsorize_threshold):
        assert self.label_scaler is not None

        labels = pd.DataFrame(
            self.label_scaler.transform(labels),
            index=labels.index,
            columns=labels.columns,
        )

        if winsorize_threshold is not None:
            labels = (
                labels.clip(-winsorize_threshold, winsorize_threshold)
                / winsorize_threshold
            )

        return labels

    def _build_label(self, rawdata_row, lookahead_window):
        # build fwd_return(window)
        pricing = rawdata_row["open"].sort_index()
        fwd_return = (
            pricing.pct_change(lookahead_window, fill_method=None)
            .shift(-lookahead_window - 1)
            .rename(f"fwd_return({lookahead_window})")
            .sort_index()
        )[: -lookahead_window - 1]

        return fwd_return

    def build_labels(self, rawdata, lookahead_window):
        labels = []
        for coin in tqdm(self.tradable_coins):
            labels.append(
                self._build_label(
                    rawdata_row=rawdata[coin], lookahead_window=lookahead_window
                ).rename(coin)
            )

        labels = pd.concat(labels, axis=1).sort_index()[self.tradable_coins]

        return labels

    def store_artifacts(
        self,
        features,
        labels,
        pricing,
        feature_scaler,
        label_scaler,
        train_ratio,
        params,
        data_store_dir,
    ):
        # Make dirs
        train_data_store_dir = os.path.join(data_store_dir, "train")
        test_data_store_dir = os.path.join(data_store_dir, "test")
        make_dirs([train_data_store_dir, test_data_store_dir])

        # Store params
        joblib.dump(feature_scaler, os.path.join(data_store_dir, "feature_scaler.pkl"))
        joblib.dump(label_scaler, os.path.join(data_store_dir, "label_scaler.pkl"))

        with open(os.path.join(data_store_dir, "dataset_params.json"), "w") as f:
            json.dump(params, f)

        print(f"[+] Metadata is stored")

        # Store dataset
        boundary_index = int(len(features.index) * train_ratio)

        for file_name, data in [
            ("X.parquet.zstd", features),
            ("Y.parquet.zstd", labels),
            ("pricing.parquet.zstd", pricing),
        ]:
            to_parquet(
                df=data.iloc[:boundary_index],
                path=os.path.join(train_data_store_dir, file_name),
            )

            to_parquet(
                df=data.iloc[boundary_index:],
                path=os.path.join(test_data_store_dir, file_name),
            )

        print(f"[+] Dataset is stored")

    def build(
        self,
        rawdata_dir=CONFIG["rawdata_dir"],
        data_store_dir=CONFIG["data_store_dir"],
        lookahead_window=CONFIG["lookahead_window"],
        train_ratio=CONFIG["train_ratio"],
        scaler_type=CONFIG["scaler_type"],
        winsorize_threshold=CONFIG["winsorize_threshold"],
        query_min_start_dt=CONFIG["query_min_start_dt"],
    ):
        assert scaler_type in ("RobustScaler", "StandardScaler")
        pandarallel.initialize()

        # Make dirs
        make_dirs([data_store_dir])

        # Set file_names
        file_names = sorted(glob(os.path.join(rawdata_dir, "*")))
        assert len(file_names) != 0

        # Build rawdata
        rawdata = self.build_rawdata(
            file_names=file_names, query_min_start_dt=query_min_start_dt
        )
        gc.collect()

        # Build features
        features, class_features = self.build_features(rawdata=rawdata)
        self.feature_scaler = self.build_scaler(data=features, scaler_type=scaler_type)
        features = self.preprocess_features(
            features=features, winsorize_threshold=winsorize_threshold
        )
        features = pd.concat([features, class_features], axis=1)[
            self.features_columns
        ].sort_index()
        gc.collect()

        # build labels
        labels = self.build_labels(rawdata=rawdata, lookahead_window=lookahead_window)
        self.label_scaler = self.build_scaler(data=labels, scaler_type=scaler_type)
        labels = self.preprocess_labels(
            labels=labels, winsorize_threshold=winsorize_threshold
        )
        gc.collect()

        # Masking with common index
        common_index = (features.index & labels.index).sort_values()
        features = features.reindex(common_index)
        labels = labels.reindex(common_index)
        pricing = rawdata.reindex(common_index)

        params = {
            "lookahead_window": lookahead_window,
            "train_ratio": train_ratio,
            "scaler_type": scaler_type,
            "features_columns": features.columns.tolist(),
            "labels_columns": labels.columns.tolist(),
            "tradable_coins": self.tradable_coins,
            "winsorize_threshold": winsorize_threshold,
            "query_min_start_dt": query_min_start_dt,
        }

        # Store Artifacts
        self.store_artifacts(
            features=features,
            labels=labels,
            pricing=pricing,
            feature_scaler=self.feature_scaler,
            label_scaler=self.label_scaler,
            train_ratio=train_ratio,
            params=params,
            data_store_dir=data_store_dir,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(DatasetBuilder)
