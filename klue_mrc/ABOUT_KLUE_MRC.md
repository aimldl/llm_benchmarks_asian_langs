# About KLUE Machine Reading Comprehension (MRC)

## Task Overview

The KLUE Machine Reading Comprehension (MRC) task evaluates a model's ability to understand Korean text and answer questions based on the given context. This is a fundamental natural language understanding task that tests reading comprehension, information extraction, and reasoning capabilities.

## Task Description

Given a Korean text passage (context) and a question, the model must:
1. **Read and understand** the provided context
2. **Identify relevant information** that answers the question
3. **Extract or generate** the appropriate answer
4. **Handle unanswerable questions** by recognizing when the context doesn't contain the answer

## Dataset Structure

Each sample in the KLUE MRC dataset contains:

- **Title**: The title of the article or document
- **Context**: A Korean text passage that serves as the reading material
- **Question**: A question in Korean that requires understanding the context
- **Answers**: A list of valid answer formulations (multiple correct answers may exist)
- **Answer Start**: Character positions where answers begin in the context (for extractive answers)
- **Is Impossible**: Boolean flag indicating if the question cannot be answered from the context

## Question Types

The MRC task includes various types of questions:

### Answerable Questions
- **Factual Questions**: "언제 이 사건이 발생했나요?" (When did this incident occur?)
- **Entity Questions**: "누가 이 프로젝트를 주도했나요?" (Who led this project?)
- **Descriptive Questions**: "이 기사의 제목은 무엇인가요?" (What is the title of this article?)
- **Reasoning Questions**: "왜 이런 결정을 내렸나요?" (Why was this decision made?)

### Unanswerable Questions
- **Missing Information**: Questions about details not mentioned in the context
- **Ambiguous Context**: Questions that cannot be definitively answered due to insufficient information
- **Contradictory Information**: Questions where the context provides conflicting information

## Evaluation Metrics

### Primary Metrics

1. **Exact Match (EM)**
   - Measures whether the predicted answer exactly matches any of the ground truth answers
   - Binary metric: 1 if exact match, 0 otherwise
   - Formula: `EM = (Number of exact matches) / (Total questions)`

2. **F1 Score**
   - Harmonic mean of precision and recall for answer prediction
   - Accounts for partial matches and different answer formulations
   - Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
   - Where:
     - Precision = (Common words between prediction and ground truth) / (Words in prediction)
     - Recall = (Common words between prediction and ground truth) / (Words in ground truth)

3. **Impossible Accuracy**
   - Accuracy on questions marked as "impossible"
   - Measures how well the model recognizes unanswerable questions
   - Formula: `Impossible Accuracy = (Correct impossible predictions) / (Total impossible questions)`

### Secondary Metrics

- **Processing Time**: Total time and average time per sample
- **Throughput**: Samples processed per second
- **Per-Type Analysis**: Separate metrics for answerable vs impossible questions

## Task Challenges

### Linguistic Challenges
- **Korean Language Complexity**: Korean has complex morphology, honorifics, and context-dependent meanings
- **Context Length**: Passages can be long and contain multiple topics
- **Answer Formulation**: Multiple valid ways to express the same answer

### Reasoning Challenges
- **Information Extraction**: Finding the exact information needed to answer the question
- **Context Understanding**: Comprehending the relationships between different parts of the text
- **Inference**: Making logical connections not explicitly stated in the text

### Technical Challenges
- **Impossible Question Detection**: Distinguishing between answerable and unanswerable questions
- **Answer Boundary Detection**: Identifying the exact span of text that constitutes the answer
- **Multiple Valid Answers**: Handling cases where different answer formulations are equally correct

## Model Requirements

To perform well on this task, a model must:

1. **Language Understanding**
   - Comprehend Korean grammar and syntax
   - Understand context and discourse structure
   - Handle various writing styles and domains

2. **Information Processing**
   - Extract relevant information from long passages
   - Identify key entities, events, and relationships
   - Filter out irrelevant information

3. **Question Answering**
   - Understand different question types and intents
   - Generate or extract appropriate answers
   - Handle multiple valid answer formulations

4. **Reasoning Capabilities**
   - Make logical inferences from the text
   - Connect information across different parts of the passage
   - Recognize when information is insufficient

## Dataset Statistics

- **Training Set**: ~18,000 samples
- **Validation Set**: ~2,000 samples
- **Test Set**: ~2,000 samples (not used in this benchmark)
- **Average Context Length**: ~500-1000 characters
- **Average Question Length**: ~20-50 characters
- **Percentage of Impossible Questions**: ~10-15%

## Related Tasks

The MRC task is related to several other natural language understanding tasks:

- **Question Answering (QA)**: General question answering without specific context
- **Information Extraction**: Extracting structured information from text
- **Text Summarization**: Creating concise summaries of longer texts
- **Reading Comprehension**: Understanding and reasoning about text content

## Applications

Machine Reading Comprehension has numerous real-world applications:

- **Search Engines**: Understanding user queries and finding relevant information
- **Virtual Assistants**: Answering questions based on available information
- **Document Analysis**: Extracting specific information from large documents
- **Educational Technology**: Creating interactive reading comprehension exercises
- **Customer Support**: Automatically answering customer questions from knowledge bases

## Benchmark Significance

The KLUE MRC benchmark is particularly important because:

1. **Korean Language Focus**: Provides evaluation for Korean language understanding
2. **Real-world Relevance**: Uses authentic Korean text from various domains
3. **Comprehensive Evaluation**: Tests both answerable and unanswerable questions
4. **Multiple Metrics**: Provides detailed performance analysis across different aspects
5. **Standardized Evaluation**: Enables fair comparison between different models

## References

- [KLUE Paper](https://arxiv.org/abs/2105.09680)
- [KLUE GitHub Repository](https://github.com/KLUE-benchmark/KLUE)
- [SQuAD Paper](https://arxiv.org/abs/1606.05250) (English MRC benchmark)
- [KorQuAD Paper](https://arxiv.org/abs/1909.07005) (Korean MRC benchmark) 