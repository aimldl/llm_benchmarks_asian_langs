# About KLUE Natural Language Inference (NLI)

## Overview

The KLUE Natural Language Inference (NLI) task is part of the Korean Language Understanding Evaluation (KLUE) benchmark suite. This task evaluates a model's ability to understand logical relationships between Korean sentences.

## Task Description

Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), is a fundamental task in natural language processing that involves determining the logical relationship between two sentences: a premise and a hypothesis.

### Input Format

- **Premise**: A statement that serves as the context or background information
- **Hypothesis**: A statement whose truth value needs to be determined relative to the premise

### Output Categories

The model must classify the relationship between the premise and hypothesis into one of three categories:

1. **Entailment (함의)**: The premise logically entails the hypothesis
   - The premise provides sufficient information to conclude that the hypothesis is true
   - Example: 
     - Premise: "김철수는 서울대학교 의과대학을 졸업했다"
     - Hypothesis: "김철수는 의사다"
     - Label: entailment

2. **Contradiction (모순)**: The premise contradicts the hypothesis
   - The premise provides information that makes the hypothesis false
   - Example:
     - Premise: "김철수는 서울대학교 의과대학을 졸업했다"
     - Hypothesis: "김철수는 의사가 아니다"
     - Label: contradiction

3. **Neutral (중립)**: The premise neither entails nor contradicts the hypothesis
   - The premise does not provide sufficient information to determine the truth value of the hypothesis
   - Example:
     - Premise: "김철수는 서울대학교 의과대학을 졸업했다"
     - Hypothesis: "오늘 날씨가 좋다"
     - Label: neutral

## Dataset Information

### KLUE-NLI Dataset

The KLUE-NLI dataset is specifically designed for Korean language understanding and contains:

- **Training set**: 24,998 sentence pairs
- **Validation set**: 3,000 sentence pairs  
- **Test set**: 3,000 sentence pairs

### Dataset Characteristics

1. **Korean Language Focus**: All sentences are in Korean, testing the model's understanding of Korean language semantics and syntax

2. **Diverse Domains**: The dataset covers various domains including:
   - News articles
   - Academic texts
   - Everyday conversations
   - Technical documents

3. **Balanced Distribution**: The dataset maintains a balanced distribution across the three labels to prevent bias in evaluation

4. **High Quality**: All sentence pairs are carefully annotated by Korean language experts

## Evaluation Metrics

### Primary Metric

- **Accuracy**: The percentage of correctly classified sentence pairs

### Additional Metrics

- **Per-label Accuracy**: Accuracy breakdown for each of the three labels (entailment, contradiction, neutral)
- **Processing Speed**: Time per sample and samples per second
- **Error Analysis**: Detailed analysis of misclassified samples

## Challenges

The KLUE-NLI task presents several challenges for language models:

1. **Korean Language Understanding**: Requires deep understanding of Korean grammar, syntax, and semantics

2. **Logical Reasoning**: Models must perform complex logical inference to determine relationships between sentences

3. **Context Sensitivity**: The same hypothesis can have different relationships with different premises depending on context

4. **Implicit Knowledge**: Some entailments require world knowledge that may not be explicitly stated

5. **Linguistic Phenomena**: Korean-specific linguistic phenomena such as honorifics, particles, and sentence endings

## Model Requirements

To perform well on the KLUE-NLI task, a model should have:

1. **Korean Language Proficiency**: Strong understanding of Korean language structure and meaning

2. **Logical Reasoning Capabilities**: Ability to perform deductive and inductive reasoning

3. **Semantic Understanding**: Deep comprehension of word meanings and sentence semantics

4. **Context Awareness**: Ability to understand how context affects meaning

5. **Robustness**: Consistent performance across different domains and writing styles

## Applications

Natural Language Inference has numerous real-world applications:

1. **Question Answering**: Determining if a candidate answer is entailed by supporting text
2. **Information Retrieval**: Finding documents that entail specific claims
3. **Text Summarization**: Ensuring summaries are entailed by source documents
4. **Fact Checking**: Verifying if claims are supported by evidence
5. **Dialogue Systems**: Understanding logical relationships in conversations
6. **Machine Translation**: Ensuring translated text maintains logical consistency

## Benchmark Implementation

This benchmark implementation:

1. **Uses Gemini 2.5 Flash**: Leverages Google's latest language model for inference
2. **Vertex AI Integration**: Utilizes Google Cloud Vertex AI for scalable model serving
3. **Comprehensive Evaluation**: Provides detailed metrics and error analysis
4. **Flexible Configuration**: Supports various model parameters and sampling options
5. **Result Export**: Saves results in multiple formats for further analysis

## References

- KLUE: Korean Language Understanding Evaluation (https://klue-benchmark.com/)
- Natural Language Inference: A Survey (Bowman et al., 2020)
- Korean Language Processing: A Comprehensive Survey (Park et al., 2021)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{klue2021,
  title={KLUE: Korean Language Understanding Evaluation},
  author={Park, Sungjoon and Kim, Seonghyeon and Lee, Seungjun and Song, Jihyun and Kim, Seungwon and Cha, Sunkyu and Oh, Dongwon and Lee, Key-Sun and Kang, Jaewoo},
  journal={arXiv preprint arXiv:2105.09680},
  year={2021}
}
``` 