import pandas as pd
from sklearn.metrics import accuracy_score

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class MinimalPairsMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )

        self.label_map = {
            "id": {"a": 0, "b": 1},
            "ta": {"a": 0, "b": 1},
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
        ].map(self.extract_response)

        # make linguistic_phenomenon a column
        self.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.inference_df["metadata"]
        ]

    def calculate_metrics(self):
        metric_dict = {"accuracy": None, "subcategories": {}}

        for phenomenon in self.inference_df["linguistic_phenomenon"].unique():
            subset = self.inference_df[
                self.inference_df["linguistic_phenomenon"] == phenomenon
            ]
            subset_predictions = subset[self.postprocessed_response_column]
            subset_references = subset[self.label_column].apply(
                lambda x: self.label_map.get(x.lower())
            )
            subset_accuracy = 100 * accuracy_score(
                y_true=subset_references, y_pred=subset_predictions
            )
            metric_dict["subcategories"].update({phenomenon: subset_accuracy})
            logger.info(f"Accuracy for phenomenon <{phenomenon}>: {subset_accuracy}")

        accuracy_list = metric_dict["subcategories"].values()

        overall_accuracy = (
            sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else None
        )

        metric_dict["accuracy"] = overall_accuracy
        logger.info(f"Overall Accuracy: {overall_accuracy}")

        metric_dict["normalized_accuracy"] = (
            self.normalize_score(overall_accuracy, 1 / len(self.label_map) * 100, 100)
            * 100
        )

        predictions = self.inference_df[self.postprocessed_response_column]
        references = self.inference_df[self.label_column].apply(
            lambda x: self.label_map.get(x.lower())
        )

        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]
        return metric_dict, self.inference_df
