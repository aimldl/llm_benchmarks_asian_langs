# TROUBLESHOOTING

This document is written in the reverse order.

## 

## Problem: ERROR - Prediction failed: Cannot get the response text

```bash
$ run custom 100
```
```bash
Running custom benchmark with 100 samples...
  ...
2025-07-06 19:00:02,828 - INFO - Starting benchmark...
2025-07-06 19:01:48,002 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 594,
    "total_token_count": 1617,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 594
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:01:40.492285Z",
  "response_id": "FMhqaP2FHqW8nvgP2IrW4As"
}
2025-07-06 19:02:56,717 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 587,
    "total_token_count": 1610,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 587
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:02:48.616387Z",
  "response_id": "WMhqaMPPJaKgnvgPiPGr4AE"
}
2025-07-06 19:03:50,104 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 589,
    "total_token_count": 1612,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 589
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:03:41.062294Z",
  "response_id": "jchqaNbmA6W8nvgP2IrW4As"
}
2025-07-06 19:05:49,089 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 590,
    "total_token_count": 1613,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 590
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:05:40.756729Z",
  "response_id": "BMlqaPmXLt3_698PqM_rwAE"
}
2025-07-06 19:07:13,267 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 591,
    "total_token_count": 1614,
    "prompt_tokens_details": [
      {
        "modality": "TEXT",
        "token_count": 591
      }
    ],
    "thoughts_token_count": 1023
  },
  "model_version": "gemini-2.5-flash",
  "create_time": "2025-07-06T19:07:05.128002Z",
  "response_id": "WclqaILoB5uCnvgP6_r5wAk"
}
  ...
Processing samples: 100%|█████████████████████| 100/100 [07:25<00:00,  4.46s/it]
2025-07-06 19:07:28,421 - INFO - Benchmark completed!
2025-07-06 19:07:28,421 - INFO - Accuracy: 0.7000 (70/100)
2025-07-06 19:07:28,421 - INFO - Total time: 445.59 seconds
2025-07-06 19:07:28,421 - INFO - Average time per sample: 4.456 seconds
  ...
Error Analysis (showing first 5 errors):
  1. True: 사회 | Predicted: 경제
     Text: 5억원 무이자 융자는 되고 7천만원 이사비는 안된다...
     Prediction: 경제

  2. True: 사회 | Predicted: 정치
     Text: 왜 수소충전소만 더 멀리 떨어져야 하나 한경연 규제개혁 건의...
     Prediction: 정치

  3. True: 사회 | Predicted: 정치
     Text: 모의선거 교육 불허 선관위·교육부 각성하라...
     Prediction: 정치

  4. True: 사회 | Predicted: 정치
     Text: 가짜뉴스 징벌적 손해배상제도 도입 변협 토론회...
     Prediction: 정치

  5. True: 사회 | Predicted: 경제
     Text: MBN 노조 부동산 물적분할 중단하고 소유경영 분리해야...
     Prediction: 경제
```
### Hint: 



## Incorrect Prompt
```bash
~/llm_benchmarks_asian_langs/klue_tc$ ./run test
Running small test with 10 samples...
project_id: vertex-workbench-notebook
2025-07-04 04:27:34,860 - INFO - Initialized Vertex AI with project: vertex-workbench-notebook, location: us-central1
2025-07-04 04:27:34,861 - INFO - Initialized model: gemini-2.5-flash
2025-07-04 04:27:34,861 - INFO - Loading KLUE YNAT dataset for topic classification...
2025-07-04 04:27:37,657 - INFO - Preparing to load a subset of 10 samples.
2025-07-04 04:27:37,659 - INFO - Reached sample limit of 10. Halting data loading.
2025-07-04 04:27:37,659 - INFO - ✅ Successfully loaded 10 samples.
2025-07-04 04:27:37,659 - INFO - Starting benchmark...
Processing samples:  70%|███████████████████████████████████               | 7/10 [00:27<00:11,  3.85s/it]2025-07-04 04:28:12,947 - ERROR - Prediction failed: Cannot get the response text.
Cannot get the Candidate text.
Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.
Content:
{
  "role": "model"
}
Candidate:
{
  "content": {
    "role": "model"
  },
  "finish_reason": "MAX_TOKENS"
}
Response:
{
  "candidates": [
    {
      "content": {
        "role": "model"
      },
      "finish_reason": "MAX_TOKENS"
    }
  ],
  "usage_metadata": {
    "prompt_token_count": 594,
    "total_token_count": 1617
  },
  "model_version": "gemini-2.5-flash"
}
Processing samples: 100%|█████████████████████████████████████████████████| 10/10 [00:45<00:00,  4.52s/it]
2025-07-04 04:28:22,840 - INFO - Benchmark completed!
2025-07-04 04:28:22,840 - INFO - Accuracy: 0.4000 (4/10)
2025-07-04 04:28:22,841 - INFO - Total time: 45.18 seconds
2025-07-04 04:28:22,841 - INFO - Average time per sample: 4.518 seconds
2025-07-04 04:28:22,841 - INFO - Metrics saved to: benchmark_results/klue_tc_metrics_20250704_042822.json
2025-07-04 04:28:22,842 - INFO - Detailed results saved to: benchmark_results/klue_tc_results_20250704_042822.json
2025-07-04 04:28:22,845 - INFO - Results saved as CSV: benchmark_results/klue_tc_results_20250704_042822.csv

============================================================
KLUE Topic Classification Benchmark Results
============================================================
Model: gemini-2.5-flash
Platform: Google Cloud Vertex AI
Project: vertex-workbench-notebook
Location: us-central1
Accuracy: 0.4000 (4/10)
Total Time: 45.18 seconds
Average Time per Sample: 4.518 seconds
Samples per Second: 0.22

Per-label Accuracy:
  IT과학: 0.0000 (0/1)
  경제: 1.0000 (2/2)
  사회: 0.3333 (2/6)
  생활문화: 0.0000 (0/1)

Error Analysis (showing first 5 errors):
  1. True: 사회 | Predicted: 경제
     Text: 5억원 무이자 융자는 되고 7천만원 이사비는 안된다...
     Prediction: 경제

  2. True: 사회 | Predicted: 정치
     Text: 왜 수소충전소만 더 멀리 떨어져야 하나 한경연 규제개혁 건의...
     Prediction: 정치

  3. True: IT과학 | Predicted: None
     Text: 항응고제 성분 코로나19에 효과…세포실험서 확인...
     Prediction: IT·과학

  4. True: 사회 | Predicted: 정치
     Text: 모의선거 교육 불허 선관위·교육부 각성하라...
     Prediction: 정치

  5. True: 생활문화 | Predicted: None
     Text: 뮤지컬 영웅 합류한 안재욱 정성화와 다른 안중근 보여줄것...
     Prediction: 생활·문화

~/llm_benchmarks_asian_langs/klue_tc$ 
```

`생활·문화` & `IT·과학` is due to an incorrect prompt.

```python
  ...
prompt = f"""역할: 당신은 다양한 한국어 텍스트의 핵심 주제를 정확하게 분석하고 분류하는 "전문 텍스트 분류 AI"입니다.

  ...

생활·문화: 예술, 대중문화(영화, 드라마, 음악), 연예, 패션, 음식, 여행, 건강, 취미, 종교, 도서 등 일상생활과 관련된 정보
  ...
IT·과학: 정보 기술(IT), 인공지능(AI), 반도체, 인터넷, 소프트웨어, 최신 과학 연구, 우주, 생명 공학 등
  ...
두 개 이상의 카테고리에 해당될 수 있는 내용일 경우, 텍스트에서 가장 비중 있게 다루는 주제를 우선적으로 선택합니다. 예를 들어, '정부의 IT 산업 육성 정책'에 대한 글이라면 '경제'나 'IT·과학'도 관련이 있지만, 정책 발표가 핵심이므로 '정치'로 분류합니다.

...

주제:"""
```
`생활·문화` & `IT·과학` -> `생활문화` & `IT과학` 