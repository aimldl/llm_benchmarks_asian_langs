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
   - Provides a stringent evaluation of the model's ability to precisely reproduce correct text spans

2. **ROUGE-W (Weighted Longest Common Subsequence-based F1)**
   - Evaluates the quality of generated answers by comparing them against reference answers
   - Assigns higher weights to consecutive matches compared to non-consecutive matches
   - Prioritizes longer, unbroken common sequences while considering the Longest Common Subsequence (LCS)
   - Formula: 
   
   `ROUGE-W F1 = 2 * (ROUGE-W Precision * ROUGE-W Recall) / (ROUGE-W Precision + ROUGE-W Recall)`

   
   - Where:
     - ROUGE-W Precision = (Weighted overlapping units) / (Number of units in generated text)
     - ROUGE-W Recall = (Weighted overlapping units) / (Number of units in reference text)

3. **LCCS-based F1 (Longest Common Consecutive Subsequence-based F1)**
   - Directly related to ROUGE-W, emphasizing consecutive matches
   - Measures the longest common consecutive subsequence between prediction and reference
   - Particularly suitable for extractive question answering where exact wording and order matter
   - Rewards accurate and unbroken sequences of words

4. **Impossible Accuracy**
   - Accuracy on questions marked as "impossible"
   - Measures how well the model recognizes unanswerable questions
   - Formula: `Impossible Accuracy = (Correct impossible predictions) / (Total impossible questions)`

### Secondary Metrics

- **Processing Time**: Total time and average time per sample
- **Throughput**: Samples processed per second
- **Per-Type Analysis**: Separate metrics for answerable vs impossible questions

### Rationale for Metric Selection

The combination of **Exact Match (EM)** and **ROUGE-W/LCCS-based F1** provides a comprehensive evaluation framework:

1. **Exact Match (EM)** offers a stringent assessment of the model's ability to precisely reproduce correct text spans, ensuring high-quality extractive answers.

2. **ROUGE-W** provides flexibility for minor variations while heavily rewarding consecutive, accurate text spans that define effective answers in extractive MRC.

3. **LCCS-based F1** emphasizes the importance of maintaining exact wording and order, which is crucial for extractive question answering in Korean.

This metric combination balances the need for precise reproduction (EM) with flexibility for minor variations (ROUGE-W), making it particularly suitable for Korean MRC tasks where both accuracy and natural language variations are important.

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
- **Precise Text Extraction**: Ensuring extracted answers maintain exact wording and order from the source text

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
   - Generate or extract appropriate answers with precise wording
   - Handle multiple valid answer formulations while maintaining accuracy
   - Ensure extracted answers maintain exact text boundaries and order

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
4. **Advanced Metrics**: Uses ROUGE-W and LCCS-based F1 for comprehensive evaluation of extractive answers
5. **Standardized Evaluation**: Enables fair comparison between different models using industry-standard metrics

## References

- [KLUE Paper](https://arxiv.org/abs/2105.09680)
- [KLUE GitHub Repository](https://github.com/KLUE-benchmark/KLUE)
- [SQuAD Paper](https://arxiv.org/abs/1606.05250) (English MRC benchmark)
- [KorQuAD Paper](https://arxiv.org/abs/1909.07005) (Korean MRC benchmark) 