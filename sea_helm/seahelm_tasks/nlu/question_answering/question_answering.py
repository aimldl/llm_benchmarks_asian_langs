import string
from collections import Counter
from typing import Any, List

import pandas as pd
from pythainlp import word_tokenize

from base_logger import get_logger
from seahelm_tasks.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class QuestionAnsweringMetric(SeaHelmMetric):
    def __init__(
        self, inference_df: pd.DataFrame, task_config: dict, task: str, lang: str
    ):
        super().__init__(
            inference_df=inference_df, task_config=task_config, task=task, lang=lang
        )
        self.regex_string = {
            "id": r"(?<=[J|j]awaban:)[\s\r\n]*.*",
            "th": r"(?<=คำตอบ:)[\s\r\n]*.*",
            "vi": r"(?<=[C|c]âu trả lời:)[\s\r\n]*.*",
            "ta": r"(?<=பதில்:)[\s\r\n]*.*",
        }[lang]

    def normalize_answer(self, s):
        """Lower text and remove punctuation and extra whitespace."""

        # Articles only apply to English
        # def remove_articles(text):
        #     return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def get_references(self) -> List[Any]:
        if self.lang == "id":
            return self.inference_df[self.label_column]
        elif self.lang in ["vi", "th", "ta"]:
            return [[x] for x in self.inference_df[self.label_column]]

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _evaluate(self, references, predictions):
        exact_match = []
        f1_list = []
        for ref, pred in zip(references, predictions):
            exact_match.append(
                self.metric_max_over_ground_truths(self.exact_match_score, pred, ref)
            )
            f1_list.append(
                self.metric_max_over_ground_truths(self._f1_score, pred, ref)
            )

        total = len(f1_list)
        exact_match = 100.0 * sum(exact_match) / total if total > 0 else None
        f1 = 100.0 * sum(f1_list) / total if total > 0 else None

        normalized_f1 = self.normalize_score(f1, 0, 1)
        results = {"exact_match": exact_match, "f1": f1, "normalized_f1": normalized_f1}
        return results, f1_list

    def _tokenize_thai_text(self, text):
        tokenized_text = " ".join(
            word_tokenize(
                engine="newmm",
                text=text,
            )
        )
        return tokenized_text

    def calculate_metrics(self):
        references = self.get_references()
        predictions = self.inference_df[self.postprocessed_response_column]

        if self.lang == "th":
            # Use PyThaiNLP newmm Thai tokenizer because Thai script does not use spaces between words
            tokenized_references = [
                [self._tokenize_thai_text(ref[0])] for ref in references
            ]
            tokenized_predictions = [
                self._tokenize_thai_text(pred) for pred in predictions
            ]

            logger.info("Tokenizing Thai text and re-evaluating...")
            results, f1_list = self._evaluate(
                tokenized_references, tokenized_predictions
            )
            logger.info(results)
        else:
            results, f1_list = self._evaluate(references, predictions)
            logger.info(results)

        self.inference_df["individual_scores"] = [
            {"normalized_f1": self.normalize_score(x, 0, 1)} for x in f1_list
        ]
        # Analyze if preds contain gold answer fully
        question_count = len(references)
        gold_in_pred = 0
        try:
            for ref, pred in zip(references, predictions):
                answer = ref[0]
                if answer.lower().strip(".") in pred.lower().strip("."):
                    gold_in_pred += 1
        except:
            logger.info(answer)
            logger.info(pred)
            raise ValueError()

        if question_count > 0:
            logger.info(
                f"{gold_in_pred} answers out of {question_count} ({gold_in_pred*100/question_count:.2f}%) can be found in the model's predictions."
            )

            results.update({"found_in_prediction": gold_in_pred * 100 / question_count})

        null_count = sum(predictions == "")
        results.update({"null_count": null_count})

        return results, self.inference_df
