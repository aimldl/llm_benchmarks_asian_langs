import pandas as pd

from base_logger import get_logger
from evaluate import load
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.scoring import BootstrapAggregator
from sacrebleu.metrics import CHRF
from seahelm_tasks.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class SummarizationMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.regex_string = {
            "id": r"(?<=[R|r]angkuman:)[\s\r\n]*.*",
            "th": r"(?<=บทสรุป:)[\s\r\n]*.*",
            "vi": r"(?<=[B|b]ản tóm tắt:)[\s\r\n]*.*",
            "ta": r"(?<=சுருக்கம்:)[\s\r\n]*.*",
            "tl": r"(?<=[B|b]uod:)[\s\r\n]*.*",
        }[lang]

        language = "thai" if self.lang == "th" else None
        self.scorer = RougeScorer(["rougeL"], use_stemmer=False, lang=language)

        self.run_chrf = task_config["use_chrf_metric"]
        self.run_bertscore = task_config["use_bertscore_metric"]
        self.run_rougeL = task_config["use_rougeL_metric"]

    def calculate_metrics(self):
        predictions = self.inference_df[self.postprocessed_response_column].to_list()
        references = self.inference_df[self.label_column].to_list()
        null_count = sum([x == "" for x in predictions])

        metric_dict = {"null_count": null_count}
        if self.run_rougeL:
            rougeL_metricx, scores = self.evaluate_with_rougeL(references, predictions)
            metric_dict.update(rougeL_metricx)
            self.inference_df["individual_scores"] = [
                {"normalized_rougel_f1": x} for x in scores
            ]

        # run chrf metrics
        if self.run_chrf:
            chrf_metrics = self.evaluate_with_chrf(references, predictions)
            metric_dict.update(chrf_metrics)

        # run bertscore metrics
        if self.run_bertscore:
            bertscore_metrics = self.evaluate_with_bertscore(references, predictions)
            metric_dict.update(bertscore_metrics)

        return metric_dict, self.inference_df

    def evaluate_with_rougeL(self, references, predictions):
        rouge_score = [
            self.scorer.score(ref, pred) for ref, pred in zip(references, predictions)
        ]

        if len(rouge_score) > 0:
            aggregator = BootstrapAggregator()

            for score in rouge_score:
                aggregator.add_scores(score)
            aggregates = aggregator.aggregate()
            mid_scores = aggregates["rougeL"].mid
            norm_f1_score = self.normalize_score(
                mid_scores.fmeasure, 0, 1
            )  # 1 is the max f1 score

            logger.info("Rouge-L Scores:")
            logger.info(
                f"Precision: {100*mid_scores.precision:.2f} | Recall: {100*mid_scores.recall:.2f} | F1: {100*mid_scores.fmeasure:.2f}"
            )

            # calculate norm score
            logger.info(f"Norm F1 Score: {100 * norm_f1_score:.2f}")

            metric_dict = {
                "rougel_precision": 100 * mid_scores.precision,
                "rougel_recall": 100 * mid_scores.recall,
                "rougel_f1": 100 * mid_scores.fmeasure,
                "normalized_rougel_f1": 100 * norm_f1_score,
            }

        normalized_scores = [
            self.normalize_score(100 * x["rougeL"].fmeasure, 0, 1) for x in rouge_score
        ]
        return metric_dict, normalized_scores

    def calculate_max_score_for_normalization(self):
        max_rouge_score = self.inference_df.apply(
            lambda x: self.scorer.score(x[self.label_column], x[self.label_column]),
            axis=1,
        )
        norm_aggregator = BootstrapAggregator()

        for score in max_rouge_score:
            norm_aggregator.add_scores(score)
        aggregates = norm_aggregator.aggregate()
        mid_scores = aggregates["rougeL"].mid

        logger.info("Normalized Rouge-L Scores:")
        logger.info(
            f"Precision: {100*mid_scores.precision:.2f} | Recall: {100*mid_scores.recall:.2f} | F1: {100*mid_scores.fmeasure:.2f}"
        )
        return mid_scores.fmeasure

    def evaluate_with_chrf(self, references, predictions):
        chrf = CHRF(word_order=2)

        if len(predictions) > 0:
            scores = chrf.corpus_score(predictions, [references])
            score = scores.score
            logger.info(f"ChrF++ Score: {score}")
        else:
            score = None

        return {
            "chrf_score": score,
            "normalized_chrf_score": self.normalize_score(score, 0, 1),
        }

    def evaluate_with_bertscore(self, references, predictions):
        bertscore = load("bertscore")

        if len(predictions) > 0:
            scores = bertscore.compute(
                predictions=predictions, references=references, lang=self.lang
            )
            score = {
                key: 100 * sum(scores[key]) / len(scores[key])
                for key in ["precision", "recall", "f1"]
            }
            logger.info(f"BERTScore F1: {score['f1']}")
        else:
            score = None

        return {
            "bertscore_precision": score["precision"],
            "bertscore_recall": score["recall"],
            "bertscore_f1": score["f1"],
            "normalized_bertscore_f1_score": self.normalize_score(score["f1"], 0, 1),
        }
