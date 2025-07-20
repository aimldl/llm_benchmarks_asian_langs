# About KLUE Sentence Textual Similarity (STS)

The KLUE Sentence Textual Similarity (STS) task is one of 8 Korean natural language understanding (NLU) tasks in the Korean Language Understanding Evaluation (KLUE) benchmark. The KLUE benchmark serves as a standard for evaluating a model's ability to comprehend and process Korean text. This task specifically focuses on measuring the semantic similarity between pairs of Korean sentences.

## Dataset: Korean Sentence Textual Similarity

The dataset used for this task consists of sentence pairs from various Korean text sources, including news articles, web content, and other Korean language resources. Each pair is annotated with a similarity score ranging from 0 to 5, where:

- **0**: Completely different meaning (no semantic similarity)
- **1**: Mostly different meaning (very low similarity)
- **2**: Partially different meaning (low similarity)
- **3**: Similar meaning (moderate similarity)
- **4**: Very similar meaning (high similarity)
- **5**: Completely identical meaning (perfect similarity)

The dataset is designed to test the model's ability to understand nuanced semantic differences in Korean language, including:
- Paraphrasing and rephrasing
- Synonym usage and lexical variations
- Grammatical differences with similar meaning
- Context-dependent interpretations
- Cultural and domain-specific expressions

For more information, read the [paper](https://arxiv.org/pdf/2105.09680)'s chapter: "3. KLUE Benchmark > 3.2 Sentence Textual Similarity (STS)"

## Evaluation Metrics

### Primary Metrics

1. **Pearson Correlation Coefficient**
   - Measures the linear correlation between predicted and true similarity scores
   - Range: [-1, 1], where 1 indicates perfect positive correlation
   - Formula: `r = Σ((x - μx)(y - μy)) / (σx * σy)`
   - Most commonly used metric for STS tasks

2. **Spearman Correlation Coefficient**
   - Measures the rank correlation between predicted and true similarity scores
   - Range: [-1, 1], where 1 indicates perfect rank correlation
   - Less sensitive to outliers than Pearson correlation
   - Captures monotonic relationships between variables

### Secondary Metrics

3. **Mean Squared Error (MSE)**
   - Average squared difference between predicted and true scores
   - Formula: `MSE = Σ(y_pred - y_true)² / n`
   - Penalizes larger errors more heavily

4. **Mean Absolute Error (MAE)**
   - Average absolute difference between predicted and true scores
   - Formula: `MAE = Σ|y_pred - y_true| / n`
   - More robust to outliers than MSE

### Rationale for Metric Selection

The combination of correlation coefficients and error metrics provides a comprehensive evaluation framework:

1. **Pearson Correlation** is the standard metric for STS tasks and measures how well the model's predictions align with human judgments on a linear scale.

2. **Spearman Correlation** captures whether the model correctly ranks sentence pairs by similarity, regardless of the absolute scale of predictions.

3. **MSE and MAE** provide direct measures of prediction accuracy and help identify if the model is systematically over- or under-predicting similarity scores.

## Technical Challenges

- **Semantic Understanding**: Distinguishing between surface-level differences and true semantic differences
- **Context Sensitivity**: Understanding how context affects meaning and similarity
- **Cultural Nuances**: Capturing Korean-specific cultural and linguistic nuances
- **Scale Calibration**: Predicting scores that align with human judgment scales
- **Ambiguity Handling**: Dealing with sentences that could have multiple valid interpretations

## Model Requirements

To perform well on this task, a model needs:

1. **Semantic Understanding**
   - Deep comprehension of Korean language semantics
   - Ability to recognize paraphrases and synonyms
   - Understanding of context-dependent meanings

2. **Similarity Assessment**
   - Capability to quantify semantic similarity on a continuous scale
   - Consistent scoring across different types of sentence pairs
   - Alignment with human judgment patterns

3. **Korean Language Proficiency**
   - Understanding of Korean grammar and syntax
   - Familiarity with Korean cultural and linguistic patterns
   - Ability to handle various Korean text styles and domains

## Benchmark Significance

The KLUE STS benchmark is important because:

1. **Semantic Understanding**: Tests fundamental language understanding capabilities
2. **Practical Applications**: Similarity scoring is used in search, recommendation, and text analysis systems
3. **Korean Language Focus**: Provides a standardized evaluation for Korean language models
4. **Continuous Scale**: Unlike classification tasks, requires fine-grained similarity assessment
5. **Human Alignment**: Measures how well model predictions align with human judgments

## Prominent Models and Techniques

The introduction of the KLUE benchmark was accompanied by the release of powerful baseline models like **KLUE-BERT** and **KLUE-RoBERTa**. These models, pre-trained on large Korean text corpora, have set high standards for performance on the STS task.

Typical approaches to solving this task involve:

1. **Fine-tuning Pre-trained Models**: Using models like BERT or RoBERTa with a regression head
2. **Siamese Networks**: Using twin networks to encode sentence pairs
3. **Cross-encoder Architectures**: Directly comparing sentence pairs in a single forward pass
4. **Ensemble Methods**: Combining predictions from multiple models

The success of these large language models highlights the effectiveness of transfer learning in Korean natural language processing, with pre-existing knowledge providing a strong foundation for semantic similarity assessment.

## Conclusion

The KLUE Sentence Textual Similarity task, with its well-defined dataset and comprehensive evaluation metrics, plays a vital role in advancing Korean NLP. It provides a standardized platform for researchers and developers to test and compare their models' semantic understanding capabilities, fostering innovation and progress in Korean language processing technology.

## Disclaimer
This content was drafted using Gemini 2.5 Pro. 