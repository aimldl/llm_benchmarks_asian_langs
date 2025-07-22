import pandas as pd

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import F1AccMetric

logger = get_logger(__name__)


class NLIMetric(F1AccMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.label_map = {
            "id": {"a": "entailment", "b": "contradiction", "c": "neutral"},
            "vi": {"a": "entailment", "b": "contradiction", "c": "neutral"},
            "th": {"a": "entailment", "b": "contradiction", "c": "neutral"},
            "ta": {"a": "entailment", "b": "contradiction", "c": "neutral"},
            "tl": {"a": "entailment", "b": "contradiction", "c": "neutral"},
        }[lang]
        self.regex_string = {
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "th": r"(?<=คำตอบ:)[\s\r\n]*.*",
            "vi": r"(?<=câu trả lời:)[\s\r\n]*.*",
            "ta": r"(?<=பதில்:)[\s\r\n]*.*",
            "tl": r"(?<=sagot:)[\s\r\n]*.*",
        }[lang]
