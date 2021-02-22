import os
import pandas as pd
from dataclasses import dataclass
from common_utils_svc import load_json
from werkzeug.utils import cached_property


@dataclass
class Config:
    @property
    def ENV(self):
        return os.environ

    @property
    def EXCHANGE_API_KEY(self):
        return self.ENV["EXCHANGE_API_KEY"]

    @property
    def EXCHANGE_SECRET_KEY(self):
        return self.ENV["EXCHANGE_SECRET_KEY"]

    @property
    def WEBHOOK_URL(self):
        return self.ENV["WEBHOOK_URL"]

    @cached_property
    def TEST_MODE(self):
        test_mode = self.ENV["TEST_MODE"]
        if test_mode in ("true", "True"):
            test_mode = True
        if test_mode in ("false", "False"):
            test_mode = False

        return test_mode

    @cached_property
    def DATASET_PARAMS(self):
        return load_json(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/dataset_params.json"
        )

    @property
    def EXP_DIR(self):
        return f"/app/dev/experiments/{self.ENV['EXP_NAME']}"

    @cached_property
    def EXP_PARAMS(self):
        return load_json(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/trainer_params.json"
        )

    @cached_property
    def EXP_MODEL_PARAMS(self):
        return self.EXP_PARAMS["model_config"]

    @cached_property
    def EXP_DATA_PARAMS(self):
        data_params = self.EXP_PARAMS["data_config"]
        data_params.pop("checkpoint_dir")
        data_params.pop("generate_output_dir")

        return data_params

    @cached_property
    def REPORT_PARAMS(self):
        return load_json(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/reports/params_{self.ENV['REPORT_PREFIX']}_{self.ENV['REPORT_BEST_INDEX']}_{self.ENV['REPORT_BASE_CURRENCY']}.json"
        )

    @cached_property
    def PREDICTION_ABS_BINS(self):
        bins = pd.read_parquet(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/generated_output/prediction_abs_bins.parquet.zstd"
        )
        bins.columns = bins.columns.map(lambda x: x.replace("-", "/"))
        return bins

    @cached_property
    def PROBABILITY_BINS(self):
        bins = pd.read_parquet(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/generated_output/probability_bins.parquet.zstd"
        )
        bins.columns = bins.columns.map(lambda x: x.replace("-", "/"))
        return bins

    @cached_property
    def TRADABLE_COINS(self):
        return [
            tradable_coin.replace("-", "/")
            for tradable_coin in self.REPORT_PARAMS["tradable_coins"]
        ]

    @property
    def BASE_CURRENCY(self):
        return self.REPORT_PARAMS["base_currency"]

    @property
    def LEVERAGE(self):
        return int(self.ENV["LEVERAGE"])


CFG = Config()
