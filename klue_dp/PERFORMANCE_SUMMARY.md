# KLUE Benchmark Performance Summary and Task Differences

This document provides a comprehensive overview of the performance metrics and key differences between the various KLUE (Korean Language Understanding Evaluation) tasks implemented in this benchmark suite.

## Overview of KLUE Tasks

The KLUE benchmark consists of 8 Korean natural language understanding tasks. This implementation covers:

1. **TC (Topic Classification)** - `klue_tc/`
2. **STS (Semantic Textual Similarity)** - `klue_sts/`
3. **NLI (Natural Language Inference)** - `klue_nli/`
4. **NER (Named Entity Recognition)** - `klue_ner/`
5. **RE (Relation Extraction)** - `klue_re/`
6. **DP (Dependency Parsing)** - `klue_dp/` ⭐ **NEW**

## Performance Metrics by Task

### 1. TC (Topic Classification)
- **Primary Metric**: Accuracy
- **Dataset**: YNAT (Yonhap News Agency Topic Classification)
- **Classes**: 7 categories (정치, 경제, 사회, 생활문화, 세계, IT과학, 스포츠)
- **Expected Performance**: 85-95% accuracy
- **Key Challenge**: Distinguishing between similar topics (e.g., 경제 vs 사회)
- **Evaluation**: Single-label classification accuracy

### 2. STS (Semantic Textual Similarity)
- **Primary Metric**: Pearson Correlation, Spearman Correlation
- **Dataset**: Korean STS dataset
- **Scale**: 0-5 similarity score
- **Expected Performance**: 0.70-0.85 correlation
- **Key Challenge**: Understanding semantic nuances in Korean
- **Evaluation**: Regression correlation with human judgments

### 3. NLI (Natural Language Inference)
- **Primary Metric**: Accuracy
- **Dataset**: Korean NLI dataset
- **Classes**: 3 (entailment, contradiction, neutral)
- **Expected Performance**: 75-85% accuracy
- **Key Challenge**: Logical reasoning in Korean context
- **Evaluation**: Three-way classification accuracy

### 4. NER (Named Entity Recognition)
- **Primary Metrics**: Precision, Recall, F1-Score
- **Dataset**: Korean NER dataset
- **Entity Types**: Person, Organization, Location, etc.
- **Expected Performance**: 70-80% F1-score
- **Key Challenge**: Korean name variations and honorifics
- **Evaluation**: Token-level entity recognition with BIO tagging

### 5. RE (Relation Extraction)
- **Primary Metrics**: Accuracy, Per-relation Accuracy
- **Dataset**: Korean RE dataset
- **Task**: Identify relationships between entity pairs
- **Expected Performance**: 60-75% accuracy
- **Key Challenge**: Complex Korean sentence structures
- **Evaluation**: Relation classification between entity pairs

### 6. DP (Dependency Parsing) ⭐ **NEW**
- **Primary Metrics**: UAS (Unlabeled Attachment Score), LAS (Labeled Attachment Score)
- **Dataset**: Korean Dependency Parsing dataset
- **Task**: Identify grammatical dependencies between words
- **Expected Performance**: 75-85% UAS, 70-80% LAS
- **Key Challenge**: Korean agglutination and particle-rich syntax
- **Evaluation**: Word-level dependency structure analysis

## Key Differences Between Tasks

### Task Complexity
1. **TC (Simplest)**: Single-label classification with clear categories
2. **STS**: Regression task requiring semantic understanding
3. **NLI**: Three-way classification with logical reasoning
4. **NER**: Token-level sequence labeling
5. **RE**: Relation classification between entities
6. **DP (Most Complex)**: Full syntactic analysis with dependency structures

### Linguistic Challenges

#### Korean-Specific Features
- **Agglutination**: DP and NER most affected by Korean's agglutinative nature
- **Particles (조사)**: DP requires understanding of Korean particles for dependency analysis
- **Honorifics**: NER and RE affected by Korean honorific system
- **Word Order**: DP must handle Korean's SOV structure and flexibility
- **Ellipsis**: All tasks affected by Korean's frequent subject/object omission

#### Task-Specific Challenges
- **TC**: Topic ambiguity, especially between 경제/사회/정치
- **STS**: Semantic nuance detection in Korean
- **NLI**: Logical reasoning in Korean cultural context
- **NER**: Korean name variations and honorifics
- **RE**: Complex sentence structures with multiple entities
- **DP**: Agglutination, particle analysis, and dependency direction

### Model Performance Expectations

#### Gemini 2.5 Flash Performance Ranges
| Task | Metric | Expected Range | Best Performance | Challenging Cases |
|------|--------|----------------|------------------|-------------------|
| TC | Accuracy | 85-95% | Simple news headlines | Ambiguous topics |
| STS | Correlation | 0.70-0.85 | Clear semantic pairs | Subtle differences |
| NLI | Accuracy | 75-85% | Simple premises | Complex reasoning |
| NER | F1-Score | 70-80% | Common entities | Rare names/honorifics |
| RE | Accuracy | 60-75% | Simple sentences | Complex structures |
| DP | UAS/LAS | 75-85%/70-80% | Simple sentences | Complex clauses |

### Prompt Engineering Requirements

#### Prompt Length and Complexity
1. **TC**: Medium-length prompts with clear category definitions
2. **STS**: Short prompts focusing on similarity judgment
3. **NLI**: Medium-length prompts with logical reasoning instructions
4. **NER**: Long prompts with entity type definitions and examples
5. **RE**: Very long prompts with relation definitions and examples
6. **DP**: Extremely long prompts with POS tags and dependency relations

#### Korean Language Considerations
- **Detailed POS Tagging**: DP requires comprehensive Korean POS tag explanations
- **Dependency Relations**: DP needs extensive dependency relation definitions
- **Entity Types**: NER and RE require Korean-specific entity type definitions
- **Cultural Context**: All tasks benefit from Korean cultural context in prompts

### Evaluation Complexity

#### Automated vs Manual Evaluation
- **TC, NLI, RE**: Fully automated evaluation
- **STS**: Automated correlation with human judgments
- **NER**: Automated token-level evaluation
- **DP**: Automated but complex dependency structure validation

#### Error Analysis Requirements
- **TC**: Per-category accuracy breakdown
- **STS**: Correlation analysis by similarity level
- **NLI**: Per-relation type analysis
- **NER**: Per-entity type analysis
- **RE**: Per-relation type analysis
- **DP**: Per-POS analysis and dependency error patterns

## Implementation Differences

### Code Structure
All tasks follow the same basic structure but with task-specific adaptations:

```python
# Common structure across all tasks
class KLUE[Task]Benchmark:
    def __init__(self, config)
    def load_dataset()
    def create_prompt()
    def predict_single()
    def calculate_metrics()
    def run_benchmark()
    def save_results()
```

### Task-Specific Adaptations

#### TC (Topic Classification)
- Simple label mapping
- Basic accuracy calculation
- Per-category performance analysis

#### STS (Semantic Textual Similarity)
- Correlation calculation
- Score normalization
- Pair-wise comparison

#### NLI (Natural Language Inference)
- Three-way classification
- Logical reasoning prompts
- Entailment analysis

#### NER (Named Entity Recognition)
- BIO tagging scheme
- Token-level evaluation
- Entity type analysis

#### RE (Relation Extraction)
- Entity pair identification
- Relation classification
- Per-relation accuracy

#### DP (Dependency Parsing)
- POS tag handling
- Dependency structure parsing
- UAS/LAS calculation
- Per-POS analysis

### Logging and Error Analysis

All tasks include:
- Professional logging with command headers
- Error extraction to separate `.err` files
- Detailed error analysis
- Performance tracking
- Intermediate result saving

## Performance Optimization Strategies

### Task-Specific Optimizations

#### TC
- Clear category definitions
- Example-based prompts
- Ambiguous case handling

#### STS
- Similarity scale explanation
- Context-aware prompts
- Cultural nuance consideration

#### NLI
- Logical reasoning instructions
- Premise-hypothesis structure
- Korean cultural context

#### NER
- Entity type definitions
- Korean name patterns
- Honorific system handling

#### RE
- Relation type definitions
- Entity pair identification
- Complex sentence handling

#### DP
- Comprehensive POS tag definitions
- Dependency relation explanations
- Korean grammar structure guidance

### Common Optimizations
- **Prompt Engineering**: Detailed, task-specific prompts
- **Error Handling**: Robust error recovery and logging
- **Performance Monitoring**: Real-time progress tracking
- **Result Analysis**: Comprehensive error analysis
- **Scalability**: Configurable sample sizes and batch processing

## Future Enhancements

### Potential Improvements
1. **Multi-task Learning**: Combined training across KLUE tasks
2. **Korean-Specific Models**: Fine-tuned models for Korean language
3. **Advanced Prompting**: Chain-of-thought and few-shot prompting
4. **Error Analysis**: Automated error pattern detection
5. **Performance Benchmarking**: Comparative analysis across models

### Additional KLUE Tasks
- **MRC (Machine Reading Comprehension)**
- **WOS (Wiki-based Open-domain Question Answering)**

## Conclusion

The KLUE benchmark suite provides a comprehensive evaluation of Korean language understanding capabilities. Each task presents unique challenges related to Korean linguistics and requires specialized prompt engineering and evaluation strategies. The DP task, being the most complex, requires the most detailed prompts and sophisticated evaluation metrics, while simpler tasks like TC can achieve higher performance with more straightforward approaches.

The consistent implementation structure across all tasks ensures maintainability and allows for easy comparison and analysis of model performance across different aspects of Korean language understanding. 