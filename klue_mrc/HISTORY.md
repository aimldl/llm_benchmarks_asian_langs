# History

- This is the summary of changes made to KLUE MRC.
- Note: The history captures only a small part of changes.


## Added Perf Metric Changed from ROUGE (-1, -2, -L) to ROUGE-W & LCCS-based F1
See the perf metric is both:
- EM (Exact Match)
- ROUGE-W

(klue) $ ./run full
  ...
     Question: 가공하는데 몇 초밖에 걸리지 않는 물질은?...
     Ground Truth: {'answer_start': [250], 'text': ['실리콘유']}
     Predicted: 무명천...
     Exact Match: 0.0000 | F1: 0.0000
     ROUGE-W: 0.0000 | LCCS-F1: 0.0000

  2. Sample ID: klue-mrc-v1_dev_00391
     Question: 실리콘을 실리콘유로 만들기 위해 거치는 과정은?...
     Ground Truth: {'answer_start': [188], 'text': ['실리콘']}
     Predicted: 중합...
     Exact Match: 0.0000 | F1: 0.0000
     ROUGE-W: 0.0000 | LCCS-F1: 0.0000

  3. Sample ID: klue-mrc-v1_dev_04030
     Question: 한국예술영재교육원 원장이 현재 학생을 가르치는 곳은?...
     Ground Truth: {'answer_start': [167, 167, 211], 'text': ['한국예술종합학교 음악원', '한국예술종합학교', '한예종']}
     Predicted: 한국예술영재교육원...
     Exact Match: 0.0000 | F1: 0.0000
     ROUGE-W: 0.0000 | LCCS-F1: 0.0000

  4. Sample ID: klue-mrc-v1_dev_00876
     Question: 마르텔리노를 죽이겠다고 협박한 시장의 국적은?...
     Ground Truth: {'answer_start': [29], 'text': ['독일']}
     Predicted: 답을 찾을 수 없습니다...
     Exact Match: 0.0000 | F1: 0.0000
     ROUGE-W: 0.0000 | LCCS-F1: 0.0000

  5. Sample ID: klue-mrc-v1_dev_00460
     Question: 튀니스의 왕이 딸의 정략결혼을 원활히 진행하기 위해 도움을 청한 사람은?...
     Ground Truth: {'answer_start': [6, 320, 320], 'text': ['귈리엘모', '귈리엘모 왕', '귈리엘모']}
     Predicted: 시칠리아의 귈리엘모 왕...
     Exact Match: 0.0000 | F1: 0.8000
     ROUGE-W: 0.5000 | LCCS-F1: 0.5000

Log files saved:
  Full output: logs/klue_mrc_full_20250715_181336.log
  Errors only: logs/klue_mrc_full_20250715_181336.err

## Improved Prompt for Better Performance
Added specific answer format rules with clear examples:

질문: "어디로 보내졌나?" → 답변: "노르웨이" (O), "노르웨이로 파견되었다" (X)
질문: "가격은?" → 답변: "79달러" (O), "79달러에 팔린다" (X)
Emphasized conciseness with explicit instructions:

"답변은 가능한 한 짧고 정확해야 합니다"
"문장을 완성하지 말고 핵심 답만 제공하세요"
The previous prompt is not specific enough about requiring short, concise answers. The model is giving verbose responses like "노르웨이로 파견되었다...." instead of just "노르웨이". The prompt was improved to be more explicit about answer format.

From

지침:
to

중요한 지침:

1. **정확한 답 찾기**: 질문에 대한 답이 지문에 명확히 나와 있는지 확인하세요.
2. **문맥 이해**: 지문의 전체적인 맥락을 파악하여 정확한 답을 찾으세요.
3. **답의 형태**: 
   - 답이 지문에 있으면: 지문에서 그대로 추출하여 답하세요
   - 답이 지문에 없으면: "답을 찾을 수 없습니다"라고 답하세요
4. **한국어 특성 고려**: 한국어의 문법과 표현을 정확히 이해하여 답하세요.
5. **명확성**: 답은 간결하고 명확해야 합니다.

**답변 형식 규칙:**
- 답변은 가능한 한 짧고 정확해야 합니다
- 문장을 완성하지 말고 핵심 답만 제공하세요
- 예시:
  - 질문: "어디로 보내졌나?" → 답변: "노르웨이" (O), "노르웨이로 파견되었다" (X)
  - 질문: "가격은?" → 답변: "79달러" (O), "79달러에 팔린다" (X)
  - 질문: "물질은?" → 답변: "실리콘유" (O), "무명천" (X)
Results:

Before fix: Exact Match: 0.7600, F1: 0.8195
After fix: Exact Match: 0.9000, F1: 0.9000
Answerable questions: 0.9412 Exact Match (up from 0.7561)

============================================================
KLUE Machine Reading Comprehension Benchmark Results
============================================================
Model: gemini-2.5-flash
Platform: Google Cloud Vertex AI
Project: vertex-workbench-notebook
Location: us-central1
Exact Match: 0.8700
F1 Score: 0.9003
Impossible Accuracy: 0.9500
Total Samples: 100
Answerable Samples: 80
Impossible Samples: 20
Total Time: 306.62 seconds
Average Time per Sample: 3.066 seconds
Samples per Second: 0.33

Answerable Questions Performance:
  Exact Match: 0.8500
  F1 Score: 0.8879
  Sample Count: 80

Impossible Questions Performance:
  Accuracy: 0.9500
  Correct: 19/20

Error Analysis (showing first 5 errors):
  1. Sample ID: klue-mrc-v1_dev_01835
     Question: 가공하는데 몇 초밖에 걸리지 않는 물질은?...
     Ground Truth: {'answer_start': [250], 'text': ['실리콘유']}
     Predicted: 무명천...
     Exact Match: 0.0000 | F1: 0.0000

  2. Sample ID: klue-mrc-v1_dev_00391
     Question: 실리콘을 실리콘유로 만들기 위해 거치는 과정은?...
     Ground Truth: {'answer_start': [188], 'text': ['실리콘']}
     Predicted: 중합...
     Exact Match: 0.0000 | F1: 0.0000

  3. Sample ID: klue-mrc-v1_dev_04030
     Question: 한국예술영재교육원 원장이 현재 학생을 가르치는 곳은?...
     Ground Truth: {'answer_start': [167, 167, 211], 'text': ['한국예술종합학교 음악원', '한국예술종합학교', '한예종']}
     Predicted: 한국예술영재교육원...
     Exact Match: 0.0000 | F1: 0.0000

  4. Sample ID: klue-mrc-v1_dev_00876
     Question: 마르텔리노를 죽이겠다고 협박한 시장의 국적은?...
     Ground Truth: {'answer_start': [29], 'text': ['독일']}
     Predicted: 답을 찾을 수 없습니다...
     Exact Match: 0.0000 | F1: 0.0000

  5. Sample ID: klue-mrc-v1_dev_00460
     Question: 튀니스의 왕이 딸의 정략결혼을 원활히 진행하기 위해 도움을 청한 사람은?...
     Ground Truth: {'answer_start': [6, 320, 320], 'text': ['귈리엘모', '귈리엘모 왕', '귈리엘모']}
     Predicted: 시칠리아의 귈리엘모 왕...
     Exact Match: 0.0000 | F1: 0.8000

Log files saved:
  Full output: logs/klue_mrc_custom_100samples_20250714_090756.log
  Errors only: logs/klue_mrc_custom_100samples_20250714_090756.err
Measure the Improved Performance
# Reattach to the `klue` session
tmux attach -t klue

# Run the target command within the `tmux session`
$ ./run full

# Detach from the Session
Ctrl+b d