# KLUE Benchmark Performance Summary

This document provides a comprehensive comparison of performance metrics and key differences across all KLUE (Korean Language Understanding Evaluation) tasks implemented in this benchmark suite.

## Task Overview

The KLUE benchmark suite includes the following tasks:

1. **TC (Topic Classification)** - Text classification into 7 topic categories
2. **STS (Semantic Textual Similarity)** - Measuring similarity between sentence pairs
3. **NLI (Natural Language Inference)** - Determining logical relationships between sentences
4. **NER (Named Entity Recognition)** - Identifying and classifying named entities
5. **RE (Relation Extraction)** - Extracting relationships between entities
6. **DP (Dependency Parsing)** - Part-of-speech tagging and dependency parsing
7. **MRC (Machine Reading Comprehension)** - Answering questions based on context
8. **DST (Dialogue State Tracking)** - Tracking intent and slot values in dialogues

## Performance Metrics Comparison

| Task | Main Metrics | Secondary Metrics | Expected Performance Range | Complexity |
|------|-------------|-------------------|---------------------------|------------|
| **TC** | Accuracy | Per-class accuracy, F1 | 85-95% | Low |
| **STS** | Pearson/Spearman correlation | MSE, MAE | 0.70-0.85 | Low |
| **NLI** | Accuracy | Per-relation accuracy | 75-85% | Medium |
| **NER** | F1 (micro/macro) | Precision, Recall, Per-entity F1 | 70-85% | Medium |
| **RE** | Accuracy | Per-relation accuracy | 65-80% | High |
| **DP** | UAS, LAS | Per-POS accuracy, Per-dependency accuracy | 75-90% | Very High |
| **MRC** | Exact Match, F1 | Impossible accuracy | 60-80% | High |
| **DST** | Intent Accuracy, Overall F1 | Requested Slots F1, Slot Values F1 | 60-75% | High |

## Task Complexity Analysis

### Prompt Engineering Complexity

| Task | Prompt Length | Special Instructions | Output Format | Complexity Level |
|------|---------------|---------------------|---------------|------------------|
| **TC** | ~300 words | 7 categories with examples | Single label | Low |
| **STS** | ~200 words | 5-point scale explanation | Numeric score | Low |
| **NLI** | ~250 words | 3 relations with examples | Single label | Low |
| **NER** | ~400 words | 7 entity types with examples | JSON format | Medium |
| **RE** | ~350 words | 30 relation types | JSON format | Medium |
| **DP** | ~800 words | 35 POS tags, 50+ dependencies | JSON format | Very High |
| **MRC** | ~500 words | Context + question format | Text answer | High |
| **DST** | ~600 words | Intent types, slot types, dialogue format | Structured format | High |

### Special Handling Requirements

| Task | Special Features | Error Handling | Output Validation |
|------|-----------------|----------------|-------------------|
| **TC** | Label mapping | Simple retry | Basic validation |
| **STS** | Score normalization | Score clamping | Range validation |
| **NLI** | Relation mapping | Simple retry | Basic validation |
| **NER** | Entity extraction | JSON parsing | Entity validation |
| **RE** | Entity + relation extraction | JSON parsing | Relation validation |
| **DP** | Complex parsing | JSON parsing | Tree validation |
| **MRC** | Answer extraction | Text parsing | Answer validation |
| **DST** | Intent + slot extraction | JSON parsing | State validation |

## Key Differences by Task

### 1. **TC (Topic Classification)**
- **Simplest task** with straightforward classification
- **7 predefined categories** with clear boundaries
- **Short prompts** with category descriptions
- **High expected accuracy** due to clear task definition

### 2. **STS (Semantic Textual Similarity)**
- **Regression task** rather than classification
- **Continuous output** (0-5 scale) requiring normalization
- **Pair-wise comparison** of sentences
- **Correlation metrics** instead of accuracy

### 3. **NLI (Natural Language Inference)**
- **Three-way classification** (entailment, contradiction, neutral)
- **Sentence pair analysis** with logical reasoning
- **Medium complexity** due to reasoning requirements
- **Clear task boundaries** but requires understanding

### 4. **NER (Named Entity Recognition)**
- **Sequence labeling task** with entity extraction
- **JSON output format** requiring parsing
- **7 entity types** with specific extraction rules
- **Medium complexity** due to structured output

### 5. **RE (Relation Extraction)**
- **Entity pair analysis** with relation classification
- **30 relation types** requiring detailed understanding
- **Complex JSON output** with entity and relation information
- **High complexity** due to multiple extraction steps

### 6. **DP (Dependency Parsing)**
- **Most complex task** with linguistic analysis
- **35 POS tags** and 50+ dependency relations
- **Tree structure output** requiring validation
- **Very high complexity** due to linguistic knowledge

### 7. **MRC (Machine Reading Comprehension)**
- **Question answering task** with context understanding
- **Text generation** rather than classification
- **Context + question format** requiring comprehension
- **High complexity** due to reasoning and generation

### 8. **DST (Dialogue State Tracking)**
- **Multi-turn dialogue analysis** with state tracking
- **Intent + slot extraction** requiring context understanding
- **Structured output** with multiple components
- **High complexity** due to dialogue understanding

## Implementation Consistency

All KLUE tasks maintain consistent implementation patterns:

### Directory Structure
```
klue_[task]/
├── klue_[task]-gemini2_5flash.py    # Main benchmark script
├── run                              # Benchmark runner script
├── setup.sh                         # Environment setup script
├── install_dependencies.sh          # Dependency installation script
├── test_setup.py                    # Environment testing script
├── get_errors.sh                    # Error analysis script
├── test_logging.sh                  # Logging test script
├── verify_scripts.sh                # Script verification script
├── requirements.txt                 # Python dependencies
├── README.md                        # Task-specific documentation
├── ABOUT_KLUE_[TASK].md            # Detailed task description
├── TROUBLESHOOTING.md               # Troubleshooting guide
├── VERTEX_AI_SETUP.md               # Vertex AI setup guide
├── logs/                            # Log files directory
├── benchmark_results/               # Benchmark results directory
├── result_analysis/                 # Error analysis results
└── eval_dataset/                    # Evaluation dataset directory
```

### Logging and Error Handling
- **Consistent log format** across all tasks
- **Error extraction** into separate .err files
- **Command headers** in log files
- **Timestamp-based naming** convention
- **Error analysis** with sample details

### Script Functionality
- **Three run modes**: test, custom, full
- **Environment verification** scripts
- **Dependency management** scripts
- **Error analysis** tools
- **Logging verification** tools

## Performance Expectations

### Easy Tasks (High Expected Performance)
- **TC**: 85-95% accuracy (clear categories, simple classification)
- **STS**: 0.70-0.85 correlation (straightforward similarity)

### Medium Tasks (Moderate Expected Performance)
- **NLI**: 75-85% accuracy (logical reasoning required)
- **NER**: 70-85% F1 (entity extraction with clear boundaries)

### Hard Tasks (Lower Expected Performance)
- **RE**: 65-80% accuracy (complex entity-relation extraction)
- **MRC**: 60-80% F1 (comprehension and generation)
- **DST**: 60-75% F1 (dialogue understanding and state tracking)

### Very Hard Tasks (Variable Performance)
- **DP**: 75-90% UAS/LAS (linguistic analysis, highly dependent on model capabilities)

## Recommendations

### For Research
1. **Start with TC/STS** for baseline performance
2. **Use NLI/NER** for medium complexity tasks
3. **Test RE/MRC/DST** for advanced capabilities
4. **Use DP** for comprehensive linguistic evaluation

### For Production
1. **TC/STS** are ready for production use
2. **NLI/NER** require careful validation
3. **RE/MRC/DST** need extensive testing
4. **DP** requires specialized evaluation

### For Development
1. **Consistent codebase** across all tasks
2. **Standardized logging** and error handling
3. **Comprehensive documentation** for each task
4. **Automated testing** and verification scripts

## Conclusion

The KLUE benchmark suite provides a comprehensive evaluation framework for Korean language understanding tasks. Each task offers different challenges and insights into model capabilities:

- **Simple tasks** (TC, STS) provide baseline performance metrics
- **Medium tasks** (NLI, NER) test reasoning and extraction abilities
- **Complex tasks** (RE, MRC, DST) evaluate advanced understanding
- **Linguistic tasks** (DP) assess deep language analysis

The consistent implementation across all tasks ensures reliable comparison and evaluation of model performance on Korean language understanding. 