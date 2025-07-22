import pandas as pd

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import F1AccMetric

logger = get_logger(__name__)


class SentimentAnalysisMetric(F1AccMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.label_map = {
            "id": {"positif": "positive", "negatif": "negative", "netral": "neutral"},
            "vi": {
                "tích cực": "positive",
                "tiêu cực": "negative",
                "trung lập": "neutral",
            },
            "th": {"แง่บวก": "positive", "แง่ลบ": "negative", "เฉยๆ": "neutral"},
            "ta": {"நேர்மறை": "positive", "எதிர்மறை": "negative"},
            "tl": {
                "positibo": "positive",
                "negatibo": "negative",
                "neutral": "neutral",
            },
            "jv": {"positif": "positive", "negatif": "negative", "netral": "neutral"},
            "su": {"positip": "positive", "negatip": "negative", "netral": "neutral"},
        }[lang]
        self.regex_string = {
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "th": r"(?<=คำตอบ:)[\s\r\n]*.*",
            "vi": r"(?<=câu trả lời:)[\s\r\n]*.*",
            "ta": r"(?<=பதில்:)[\s\r\n]*.*",
            "tl": r"(?<=sagot:)[\s\r\n]*.*",
            "jv": r"(?<=jawaban:)[\s\r\n]*.*",
            "su": r"(?<=jawaban:)[\s\r\n]*.*",
        }[lang]
