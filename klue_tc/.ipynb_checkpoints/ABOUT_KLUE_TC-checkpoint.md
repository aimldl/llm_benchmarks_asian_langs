# About KLUE Topic Classification

The KLUE Topic Classification task is one of 8 Korean natural language understanding (NLU) tasks in the Korean Language Understanding Evaluation (KLUE) benchmark. The KLUE benchmark serves as a standard for evaluating a model's ability to comprehend and categorize Korean text. This task specifically focuses on classifying the topic of a given news headline from the Yonhap News Agency.

## Dataset: Yonhap News Agency Topic Classification (YNAT)

The dataset used for this task is the **Yonhap News Agency Topic Classification (YNAT)** dataset. It is a collection of news headlines from the Yonhap News Agency, meticulously curated for the purpose of topic classification. The dataset is divided into training, validation, and test sets.

A key characteristic of the YNAT dataset is its seven distinct topic categories:

* Politics (정치)
* Economy (경제)
* Society (사회)
* Culture (생활문화)
* World (세계)
* IT/Science (IT과학)
* Sports (스포츠)

This diverse range of topics requires a nuanced understanding of Korean vocabulary and context to achieve high performance.

For more information, 
- read the [paper](https://arxiv.org/pdf/2105.09680 )'s chapter: "3. KLUE Benchmark > 3.1 Topic Classification (TC)" and
- see the raw JSON file [klue_benchmark/ynat-v1.1/ynat-v1.1_dev.json](https://raw.githubusercontent.com/KLUE-benchmark/KLUE/refs/heads/main/klue_benchmark/ynat-v1.1/ynat-v1.1_dev.json)

The first two samples in the JSON file are below.
```JSON
[
    {
        "guid": "ynat-v1_dev_00000",
        "title": "5억원 무이자 융자는 되고 7천만원 이사비는 안된다",
        "predefined_news_category": "경제",
        "label": "사회",
        "annotations": {
            "annotators": [
                "18",
                "03",
                "15"
            ],
            "annotations": {
                "first-scope": [
                    "사회",
                    "사회",
                    "경제"
                ],
                "second-scope": [
                    "해당없음",
                    "해당없음",
                    "사회"
                ],
                "third-scope": [
                    "해당없음",
                    "해당없음",
                    "생활문화"
                ]
            }
        },
        "url": "https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=101&sid2=260&oid=001&aid=0009563542",
        "date": "2017.09.21. 오후 5:09"
    },
    {
        "guid": "ynat-v1_dev_00001",
        "title": "왜 수소충전소만 더 멀리 떨어져야 하나 한경연 규제개혁 건의",
        "predefined_news_category": "경제",
        "label": "사회",
        "annotations": {
            "annotators": [
                "11",
                "14",
                "13"
            ],
            "annotations": {
                "first-scope": [
                    "정치",
                    "사회",
                    "사회"
                ],
                "second-scope": [
                    "사회",
                    "해당없음",
                    "해당없음"
                ],
                "third-scope": [
                    "경제",
                    "해당없음",
                    "해당없음"
                ]
            }
        },
        "url": "https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=101&sid2=261&oid=001&aid=0010476386",
        "date": "2018.11.19. 오전 11:53"
    },
    ...
  ```

## Reformatted Dataset for Quick Review
The essential part of the JSON file was reformatted for a quick review using Gemini 2.5 Pro. The prompt is below.

Note: Only up to `ynat-v1_dev_01109` was covered by Gemini's context window size when I copied and pasted the entire file.

```
I have a long JSON file in the attached file which was taken from https://raw.githubusercontent.com/KLUE-benchmark/KLUE/refs/heads/main/klue_benchmark/ynat-v1.1/ynat-v1.1_dev.json. 

Extract only four keys (guid, title, predefined_news_category, label) in the order of "guid, predefined_news_category, label, title" and reformat it into a .csv format. For example, the following part:
---

    {

        "guid": "ynat-v1_dev_00000",

        "title": "5억원 무이자 융자는 되고 7천만원 이사비는 안된다",

        "predefined_news_category": "경제",

        "label": "사회",

        "annotations": {

            "annotators": [

                "18",

                "03",

                "15"

            ],

            "annotations": {

                "first-scope": [

                    "사회",

                    "사회",

                    "경제"

                ],

                "second-scope": [

                    "해당없음",

                    "해당없음",

                    "사회"

                ],

                "third-scope": [

                    "해당없음",

                    "해당없음",

                    "생활문화"

                ]

            }

        },

        "url": "https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=101&sid2=260&oid=001&aid=0009563542",

        "date": "2017.09.21. 오후 5:09"

    },

    {

        "guid": "ynat-v1_dev_00001",

        "title": "왜 수소충전소만 더 멀리 떨어져야 하나 한경연 규제개혁 건의",

        "predefined_news_category": "경제",

        "label": "사회",

        "annotations": {

            "annotators": [

                "11",

                "14",

                "13"

            ],

            "annotations": {

                "first-scope": [

                    "정치",

                    "사회",

                    "사회"

                ],

                "second-scope": [

                    "사회",

                    "해당없음",

                    "해당없음"

                ],

                "third-scope": [

                    "경제",

                    "해당없음",

                    "해당없음"

                ]

            }

        },

        "url": "https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=101&sid2=261&oid=001&aid=0010476386",

        "date": "2018.11.19. 오전 11:53"

    },

---
should be formatted as follows:
---
00000, 경제, 사회, 5억원 무이자 융자는 되고 7천만원 이사비는 안된다
00001, 경제, 사회, 왜 수소충전소만 더 멀리 떨어져야 하나 한경연 규제개혁 건의
---
For "guid", the prefix of "ynat-v1_dev_" was removed from "ynat-v1_dev_00000"
```

Gemini's answer:
```
 have successfully extracted and reformatted the data from your JSON file.

 ynat-v1.1_..._extracted 
CSV

Here's what I did:
Loaded the data: The ynat-v1.1_dev.json file was loaded.

Extracted keys: I extracted the guid, title, predefined_news_category, and label keys.

Reordered columns: The data was reordered to "guid, predefined_news_category, label, title".

Cleaned 'guid': The "ynat-v1_dev_" prefix was removed from the guid values.

Saved as CSV: The final result has been saved into a new CSV file.

You can download the newly created file using the link below:

ynat-v1.1_dev_extracted.csv
```

## Evaluation Metric: Macro F1 Score

The primary evaluation metric for the KLUE Topic Classification task is the **Macro F1 Score**. This metric is the unweighted mean of the F1 scores for each of the seven topic categories. The choice of Macro F1 ensures that the performance on less frequent categories is given equal importance as the performance on more common ones, providing a more balanced assessment of the model's capabilities across all topics.

## Prominent Models and Techniques

The introduction of the KLUE benchmark was accompanied by the release of two powerful baseline models, **KLUE-BERT** and **KLUE-RoBERTa**. These models, which are pre-trained on a large corpus of Korean text, have set a high standard for performance on the Topic Classification task.

The typical approach to solving this task involves fine-tuning these pre-trained language models on the YNAT dataset. The process generally includes:

1.  **Tokenization:** The news headlines are tokenized using a tokenizer that is compatible with the pre-trained model (e.g., a subword tokenizer like WordPiece or SentencePiece).
2.  **Input Formatting:** The tokenized headlines are then formatted into the input structure expected by the model, often with special tokens like `[CLS]` and `[SEP]`.
3.  **Fine-tuning:** The pre-trained model is then fine-tuned on the labeled YNAT training data. This involves adding a classification layer on top of the pre-trained model and training the entire model to minimize a loss function, such as cross-entropy loss.

The success of these large language models highlights the effectiveness of transfer learning in the domain of Korean natural language processing. The models' pre-existing knowledge of the Korean language provides a strong foundation for the more specific task of topic classification.

In conclusion, the KLUE Topic Classification task, with its well-defined dataset and rigorous evaluation metric, plays a vital role in advancing the field of Korean NLP. It provides a standardized platform for researchers and developers to test and compare their models, fostering innovation and progress in understanding and processing the Korean language.

## Disclaimer
This content was drafted using Gemini Gemini 2.5 Pro.

```
Tell me about the KLUE Topic Classification task. And could you format this output in Markdown for a klue-tc.md file?
```