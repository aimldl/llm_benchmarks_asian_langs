import pandas as pd

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import F1AccMetric

logger = get_logger(__name__)


class MetaphorUnderstandingMetric(F1AccMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.label_map = {
            "id": {"a": 0, "b": 1},
            "jv": {"a": 0, "b": 1},
            "su": {"a": 0, "b": 1},
        }[lang]
        self.regex_string = {
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "jv": r"(?<=jawaban:)[\s\r\n]*.*",
            "su": r"(?<=jawaban:)[\s\r\n]*.*",
        }[lang]
