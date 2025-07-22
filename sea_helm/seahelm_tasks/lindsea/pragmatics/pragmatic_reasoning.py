import re

import pandas as pd
from sklearn.metrics import accuracy_score

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class PragmaticReasoningSingleSentenceMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.null_label = task_config["null_label"]

        def _clean_id_pragmatic_reasoning_single_sentence_response(
            response, choices_translated
        ) -> int:
            if choices_translated == "Benar atau Salah":
                return {"benar": 1, "salah": 0}.get(response, self.null_label)
            elif choices_translated == "Ya atau Tidak":
                return {"ya": 1, "tidak": 0}.get(response, self.null_label)
            else:
                logger.warning("Choices for this instance is invalid.")
                return 2

        def _clean_ta_pragmatic_reasoning_single_sentence_response(
            response, choices_translated
        ) -> int:
            return {"உண்மை": 1, "பொய்": 0}.get(response, self.null_label)

        self.label_map = {
            "id": _clean_id_pragmatic_reasoning_single_sentence_response,
            "ta": _clean_ta_pragmatic_reasoning_single_sentence_response,
        }[lang]
        self.regex_string = {
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "ta": r"(?<=பதில்:)[\s\r\n]*.*",
        }[lang]

    def extract_response(self, row):
        # try to extract the answer from the response using regex else return the response as it is
        response = row[self.response_column][0].lower()
        try:
            output = re.search(self.regex_string, response).group(0)
            output = output.strip("$")
        except:
            output = response

        output = self.normalize_answer(output)
        output = self.label_map(output, row["prompts"][0]["choices_translated"])
        return output

    def postprocess_responses(self):
        self.inference_df[self.postprocessed_response_column] = self.inference_df.apply(
            self.extract_response,
            axis=1,
        )

        # make linguistic_phenomenon a column
        self.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.inference_df["metadata"]
        ]

    def calculate_metrics(self):
        metric_dict = {"subcategories": {}}
        for phenomenon in self.inference_df["linguistic_phenomenon"].unique():
            subset = self.inference_df[
                self.inference_df["linguistic_phenomenon"] == phenomenon
            ]
            subset_predictions = subset[self.postprocessed_response_column]
            subset_references = subset[self.label_column].replace(
                {"True": 1, "False": 0, "Yes": 1, "No": 0, True: 1, False: 0}
            )
            subset_correct = accuracy_score(
                y_true=subset_references, y_pred=subset_predictions, normalize=False
            )
            subset_size = len(subset_references)

            metric_dict["subcategories"].update(
                {phenomenon: (subset_correct, subset_size)}
            )
            logger.info(
                f"Accuracy for phenomenon <{phenomenon}>: {subset_correct} / {subset_size}"
            )

        predictions = self.inference_df[self.postprocessed_response_column]
        references = self.inference_df[self.label_column].replace(
            {"True": 1, "False": 0, "Yes": 1, "No": 0, True: 1, False: 0}
        )
        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]
        return metric_dict, self.inference_df


class PragmaticReasoningSentencePairMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.label_map = {
            "id": {"benar": 1, "salah": 0},
            "ta": {"உண்மை": 1, "பொய்": 0},
        }[lang]
        self.regex_string = {
            "id": r"(?<=jawaban:)[\s\r\n]*.*",
            "ta": r"(?<=பதில்:)[\s\r\n]*.*",
        }[lang]
        self.null_label = task_config["null_label"]

    def extract_response(self, response):
        output = super().extract_response(response, use_lowercase=True)
        output = self.normalize_answer(output)
        output = self.label_map.get(output, self.null_label)
        return output

    def postprocess_responses(self):
        self.inference_df[self.postprocessed_response_column] = self.inference_df[
            self.response_column
        ].apply(self.extract_response)

        # make linguistic_phenomenon a column
        self.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.inference_df["metadata"]
        ]

    def calculate_metrics(self):
        metric_dict = {"subcategories": {}}
        for phenomenon in self.inference_df["linguistic_phenomenon"].unique():
            subset = self.inference_df[
                self.inference_df["linguistic_phenomenon"] == phenomenon
            ]
            subset_predictions = subset[self.postprocessed_response_column]
            subset_references = subset[self.label_column].replace({True: 1, False: 0})
            subset_correct = accuracy_score(
                y_true=subset_references, y_pred=subset_predictions, normalize=False
            )
            subset_size = len(subset_references)

            metric_dict["subcategories"].update(
                {phenomenon: (subset_correct, subset_size)}
            )
            logger.info(
                f"Accuracy for phenomenon <{phenomenon}>: {subset_correct} / {subset_size}"
            )

        predictions = self.inference_df[self.postprocessed_response_column]
        references = self.inference_df[self.label_column].replace({True: 1, False: 0})
        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]

        return metric_dict, self.inference_df
