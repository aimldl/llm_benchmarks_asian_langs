import pandas as pd

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import F1AccMetric

logger = get_logger(__name__)


class QuestionAnsweringMultipleChoiceMetric(F1AccMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )

        self.label_map = {
            "en": {"a": 0, "b": 1, "c": 2, "d": 3},
            "id": {"a": 0, "b": 1, "c": 2, "d": 3},
            "km": {"a": 0, "b": 1, "c": 2, "d": 3},
            "lo": {"a": 0, "b": 1, "c": 2, "d": 3},
            "my": {"a": 0, "b": 1, "c": 2, "d": 3},
            "ta": {"a": 0, "b": 1, "c": 2, "d": 3},
            "tl": {"a": 0, "b": 1, "c": 2, "d": 3},
            "th": {"a": 0, "b": 1, "c": 2, "d": 3},
            "vi": {"a": 0, "b": 1, "c": 2, "d": 3},
            "zh-s": {"a": 0, "b": 1, "c": 2, "d": 3},
            "zh-t": {"a": 0, "b": 1, "c": 2, "d": 3},
            "ms": {"a": 0, "b": 1, "c": 2, "d": 3},
        }[lang]
        self.regex_string = {
            "en": r"(?<=answer:)[\s\r\n]*.*",
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "km": r"(?<=ចម្លើយ[:|៖])[\s\r\n]*.*",
            "lo": r"(?<=ຄຳຕອບ:)[\s\r\n]*.*",
            "my": r"(?<=အဖြေ:)[\s\r\n]*.*",
            "ta": r"(?<=பதில்:)[\s\r\n]*.*",
            "tl": r"(?<=sagot:)[\s\r\n]*.*",
            "th": r"(?<=คำตอบ:)[\s\r\n]*.*",
            "vi": r"(?<=câu trả lời:)[\s\r\n]*.*",
            "zh-s": r"(?<=答案[:|：])[\s\r\n]*.*",
            "zh-t": r"(?<=答案[:|：])[\s\r\n]*.*",
            "ms": r"(?<=jawapan:)[\s\r\n]*.*",  # UNVERIFIED PROMPT AND TEMPLATE
        }[lang]
