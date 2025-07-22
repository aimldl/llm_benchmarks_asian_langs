import pandas as pd

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import F1AccMetric

logger = get_logger(__name__)


class KalahiMCMetric(F1AccMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )

        self.label_map = {
            "tl": {"a": "A", "b": "B", "c": "C", "d": "D"},
        }[lang]
        self.regex_string = {
            "tl": r"(?<=sagot:)[\s\r\n]*.*",
        }[lang]
