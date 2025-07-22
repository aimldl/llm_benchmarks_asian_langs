import pandas as pd

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import F1AccMetric

logger = get_logger(__name__)


class ToxicityDetectionMetric(F1AccMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):

        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.label_map = {
            "id": {"bersih": 0, "kasar": 1, "benci": 2},
            "th": {"y": 1, "n": 0},
            "tl": {"malinis": 0, "mapoot": 1},
            "vi": {"sạch": 0, "công kích": 1, "thù ghét": 2},
        }[lang]
        self.regex_string = {
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "th": r"(?<=คำตอบ:)[\s\r\n]*.*",
            "tl": r"(?<=sagot:)[\s\r\n]*.*",
            "vi": r"(?<=câu trả lời:)[\s\r\n]*.*",
        }[lang]
