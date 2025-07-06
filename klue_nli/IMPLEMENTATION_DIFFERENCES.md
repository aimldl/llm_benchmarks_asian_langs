# Implementation Differences: KLUE-TC vs KLUE-NLI

This document outlines the key differences between the KLUE Topic Classification (TC) and KLUE Natural Language Inference (NLI) benchmark implementations.

## Core Task Differences

### KLUE-TC (Topic Classification)
- **Input**: Single text (news article title)
- **Output**: One of 7 topic categories (정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠)
- **Dataset**: KLUE YNAT dataset
- **Model Task**: Classify the main topic of a given text

### KLUE-NLI (Natural Language Inference)
- **Input**: Two texts (premise and hypothesis)
- **Output**: One of 3 logical relationships (entailment, contradiction, neutral)
- **Dataset**: KLUE NLI dataset
- **Model Task**: Determine the logical relationship between two sentences

## Code Adaptations

### 1. Main Script (`klue_nli-gemini2_5flash.py`)

#### Class Name Change
```python
# TC Version
class KLUETopicClassificationBenchmark:

# NLI Version  
class KLUENaturalLanguageInferenceBenchmark:
```

#### Label Mapping
```python
# TC Version - 7 topic categories
LABEL_MAP = {
    0: "IT과학", 1: "경제", 2: "사회", 3: "생활문화", 
    4: "세계", 5: "스포츠", 6: "정치"
}

# NLI Version - 3 logical relationships
LABEL_MAP = {
    0: "entailment",      # 함의
    1: "contradiction",   # 모순  
    2: "neutral"          # 중립
}
```

#### Dataset Loading
```python
# TC Version
validation_dataset = load_dataset('klue', 'ynat', split='validation')
processed_data.append({
    "id": item["guid"],
    "text": item["title"],
    "label": item["label"],
    "label_text": self.LABEL_MAP.get(item["label"], "Unknown Label")
})

# NLI Version
validation_dataset = load_dataset('klue', 'nli', split='validation')
processed_data.append({
    "id": item["guid"],
    "premise": item["premise"],
    "hypothesis": item["hypothesis"],
    "label": item["label"],
    "label_text": self.LABEL_MAP.get(item["label"], "Unknown Label")
})
```

#### Prompt Engineering
```python
# TC Version - Topic classification prompt
def create_prompt(self, text: str) -> str:
    prompt = f"""역할: 당신은 다양한 한국어 텍스트의 핵심 주제를 정확하게 분석하고 분류하는 "전문 텍스트 분류 AI"입니다.
    임무: 아래에 제시된 텍스트의 핵심 내용을 파악하여, 가장 적합한 주제 카테고리 하나를 선택해 주세요.
    주제 카테고리: 정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠
    텍스트: {text}
    주제:"""

# NLI Version - Natural language inference prompt
def create_prompt(self, premise: str, hypothesis: str) -> str:
    prompt = f"""역할: 당신은 한국어 자연어 추론(Natural Language Inference)을 수행하는 "전문 언어 분석 AI"입니다.
    임무: 주어진 전제(premise)와 가설(hypothesis) 사이의 논리적 관계를 분석하여 다음 세 가지 중 하나로 분류해 주세요.
    분류 기준: entailment (함의), contradiction (모순), neutral (중립)
    전제: {premise}
    가설: {hypothesis}
    관계:"""
```

#### Prediction Method
```python
# TC Version
def predict_single(self, text: str) -> Dict[str, Any]:

# NLI Version
def predict_single(self, premise: str, hypothesis: str) -> Dict[str, Any]:
```

#### Benchmark Execution
```python
# TC Version
prediction = self.predict_single(item["text"])
result = {
    "id": item["id"],
    "text": item["text"],
    "true_label": item["label"],
    # ...
}

# NLI Version
prediction = self.predict_single(item["premise"], item["hypothesis"])
result = {
    "id": item["id"],
    "premise": item["premise"],
    "hypothesis": item["hypothesis"],
    "true_label": item["label"],
    # ...
}
```

### 2. Test Script (`test_setup.py`)

#### Dataset Testing
```python
# TC Version
dataset = load_dataset("klue", "tc")
sample = dataset['test'][0]
print(f"  - Sample text: {sample['title']} {sample['text'][:100]}...")
print(f"  - Sample label: {sample['label']}")

# NLI Version
dataset = load_dataset("klue", "nli")
sample = dataset['test'][0]
print(f"  - Sample premise: {sample['premise'][:100]}...")
print(f"  - Sample hypothesis: {sample['hypothesis'][:100]}...")
print(f"  - Sample label: {sample['label']}")
```

### 3. Documentation Updates

#### README.md
- Updated task description from topic classification to natural language inference
- Changed examples and explanations to reflect NLI task
- Updated command-line examples with NLI-specific terminology

#### ABOUT_KLUE_NLI.md
- New file explaining the NLI task in detail
- Provides examples of entailment, contradiction, and neutral relationships
- Describes the KLUE-NLI dataset characteristics

## Key Technical Differences

### 1. Input Processing
- **TC**: Processes single text input
- **NLI**: Processes paired text input (premise + hypothesis)

### 2. Output Parsing
- **TC**: Maps Korean topic names to label IDs
- **NLI**: Maps English relationship terms to label IDs with fallback to Korean terms

### 3. Prompt Complexity
- **TC**: Relatively simple classification prompt
- **NLI**: More complex prompt explaining logical relationships with examples

### 4. Error Handling
- **TC**: Basic label matching
- **NLI**: Enhanced label matching with partial matches and Korean term fallbacks

## File Structure Comparison

Both implementations maintain the same file structure:

```
klue_tc/                    klue_nli/
├── klue_tc-gemini2_5flash.py    ├── klue_nli-gemini2_5flash.py
├── requirements.txt              ├── requirements.txt
├── README.md                     ├── README.md
├── setup.sh                      ├── setup.sh
├── run                           ├── run
├── test_setup.py                 ├── test_setup.py
├── install_dependencies.sh       ├── install_dependencies.sh
├── verify_scripts.sh             ├── verify_scripts.sh
├── ABOUT_KLUE_TC.md              ├── ABOUT_KLUE_NLI.md
└── VERTEX_AI_SETUP.md            └── VERTEX_AI_SETUP.md
```

## Usage Differences

### Command Line
```bash
# TC Version
python klue_tc-gemini2_5flash.py --project-id "your-project-id"

# NLI Version
python klue_nli-gemini2_5flash.py --project-id "your-project-id"
```

### Run Script
```bash
# TC Version
./run test    # Runs TC benchmark with 10 samples

# NLI Version  
./run test    # Runs NLI benchmark with 10 samples
```

## Performance Considerations

### 1. Input Length
- **TC**: Typically shorter inputs (news titles)
- **NLI**: Longer inputs (premise + hypothesis pairs)

### 2. Token Usage
- **TC**: Lower token consumption per sample
- **NLI**: Higher token consumption due to paired inputs

### 3. Processing Time
- **TC**: Faster processing due to simpler task
- **NLI**: Slower processing due to complex logical reasoning

## Extensibility

The modular design allows easy adaptation to other KLUE tasks:

1. **Change dataset**: Update `load_dataset()` call
2. **Update labels**: Modify `LABEL_MAP` and `REVERSE_LABEL_MAP`
3. **Adapt prompts**: Rewrite `create_prompt()` method
4. **Modify processing**: Update input/output handling in `predict_single()`
5. **Update documentation**: Modify README and other docs

This pattern can be applied to other KLUE tasks such as:
- Named Entity Recognition (NER)
- Semantic Textual Similarity (STS)
- Question Answering (QA)
- Dialogue State Tracking (DST) 