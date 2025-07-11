# About KLUE Named Entity Recognition (NER)

## Task Overview

Named Entity Recognition (NER) is a fundamental natural language processing task that involves identifying and classifying named entities in text. In the context of Korean language processing, KLUE NER evaluates the ability of language models to recognize and categorize various types of named entities in Korean text.

## Entity Types

The KLUE NER task defines six main entity types:

### 1. Person (PS) - 인물
**Definition**: Names of people, including given names, family names, nicknames, titles, and honorifics.

**Examples**:
- Personal names: 김철수, 박영희, 이민수
- Titles: 대통령, 교수님, 사장님
- Nicknames: 철수, 영희
- Honorifics: 선생님, 의사님

**Characteristics**:
- Can include Korean and foreign names
- May include titles and honorifics
- Can be abbreviated or full names
- Often appear with context clues

### 2. Location (LC) - 지명
**Definition**: Names of places, including countries, cities, regions, buildings, landmarks, and geographical features.

**Examples**:
- Countries: 한국, 미국, 일본, 중국
- Cities: 서울, 부산, 대구, 인천
- Districts: 강남구, 서초구, 마포구
- Buildings: 롯데월드타워, 63빌딩, 서울타워
- Landmarks: 남산타워, 한강, 제주도

**Characteristics**:
- Can be hierarchical (country → city → district)
- May include building names and addresses
- Can be natural or man-made locations
- Often appear with directional or descriptive words

### 3. Organization (OG) - 기관
**Definition**: Names of organizations, including companies, government agencies, educational institutions, and other formal groups.

**Examples**:
- Companies: 삼성전자, LG전자, 현대자동차
- Government: 국회, 정부, 검찰청
- Schools: 서울대학교, 연세대학교, 고려대학교
- Agencies: 한국은행, 중앙일보, KBS

**Characteristics**:
- Often include organizational suffixes (회사, 학교, 기관)
- Can be abbreviated forms
- May include hierarchical structures
- Often appear in business or news contexts

### 4. Date (DT) - 날짜
**Definition**: Temporal expressions related to dates, including years, months, days, weekdays, and holidays.

**Examples**:
- Years: 2024년, 2023년, 2020년대
- Months: 3월, 12월, 다음 달
- Days: 15일, 오늘, 어제, 내일
- Weekdays: 월요일, 화요일, 주말
- Holidays: 크리스마스, 설날, 추석

**Characteristics**:
- Can be absolute or relative dates
- May include Korean calendar terms
- Often appear with temporal markers
- Can be expressed in various formats

### 5. Time (TI) - 시간
**Definition**: Time expressions, including hours, minutes, seconds, and time periods.

**Examples**:
- Hours: 오후 3시, 새벽 2시, 저녁 7시
- Minutes: 30분, 15분 전, 10분 후
- Time periods: 아침, 점심, 저녁, 밤
- Duration: 한 시간, 2시간, 반나절

**Characteristics**:
- Can be specific times or time periods
- May include relative time expressions
- Often appear with temporal context
- Can be expressed in 12 or 24-hour format

### 6. Quantity (QT) - 수량
**Definition**: Numerical expressions, including amounts, measurements, ratios, and counts.

**Examples**:
- Numbers: 100개, 5천만원, 50%
- Measurements: 1킬로그램, 2미터, 3리터
- Ratios: 3배, 절반, 10분의 1
- Counts: 10명, 5대, 3개

**Characteristics**:
- Can include various units of measurement
- May be expressed in Korean or Arabic numerals
- Often appear with quantifying words
- Can be exact or approximate values

## Dataset Information

### Source
The KLUE NER dataset is part of the Korean Language Understanding Evaluation (KLUE) benchmark suite, designed to evaluate Korean language models across various NLP tasks.

### Format
The dataset uses the BIO (Beginning-Inside-Outside) tagging scheme:

- **B-**: Beginning of an entity
- **I-**: Inside/continuation of an entity
- **O**: Outside/not part of any entity

### Example
```
Text:    김철수는 서울대학교에서 공부하고 있다
Tokens:  김철수 는 서울대학교 에서 공부하고 있다
Tags:    B-PS  O   B-OG     I-OG O   O       O
```

### Statistics
- **Training set**: ~20,000 sentences
- **Validation set**: ~1,000 sentences
- **Test set**: ~1,000 sentences (labels not provided)
- **Entity distribution**: Varies by entity type, with organizations and locations being most common

## Evaluation Metrics

### Primary Metrics
1. **Precision**: Number of correctly identified entities / Number of predicted entities
2. **Recall**: Number of correctly identified entities / Number of true entities
3. **F1 Score**: Harmonic mean of precision and recall

### Secondary Metrics
- **Per-entity Type Performance**: Individual scores for each entity type
- **Processing Time**: Total and per-sample processing times
- **Error Analysis**: Detailed breakdown of common error patterns

## Challenges in Korean NER

### Linguistic Challenges
1. **Agglutinative Nature**: Korean is an agglutinative language where morphemes are attached to stems
2. **No Spaces**: Korean text doesn't use spaces between words, making tokenization crucial
3. **Honorifics**: Complex honorific system affects entity recognition
4. **Context Dependency**: Entity boundaries often depend on context

### Technical Challenges
1. **Tokenization**: Proper tokenization is essential for accurate NER
2. **Entity Boundaries**: Determining exact entity boundaries can be ambiguous
3. **Entity Type Ambiguity**: Some entities can belong to multiple categories
4. **Rare Entities**: Handling rare or unseen entity names

## Model Performance Considerations

### Prompt Engineering
The benchmark uses detailed prompts that include:
- Clear role definition for the AI
- Comprehensive entity type definitions with examples
- Structured output format requirements
- Context handling guidelines

### Safety and Reliability
- Safety filters are disabled to maximize performance
- Error handling for API failures and timeouts
- Comprehensive logging for debugging
- Intermediate result saving for long runs

### Performance Optimization
- Rate limiting to avoid API quotas
- Efficient data processing
- Memory management for large datasets
- Parallel processing considerations

## Applications

Korean NER has numerous practical applications:

### Information Extraction
- News article analysis
- Document processing
- Social media monitoring
- Legal document analysis

### Search and Recommendation
- Entity-based search
- Content recommendation
- Knowledge graph construction
- Question answering systems

### Business Intelligence
- Market analysis
- Competitor monitoring
- Customer feedback analysis
- Risk assessment

### Academic Research
- Literature analysis
- Citation analysis
- Research trend identification
- Academic network analysis

## Related Tasks

KLUE NER is related to other NLP tasks:

1. **Named Entity Linking (NEL)**: Connecting entities to knowledge bases
2. **Relation Extraction**: Identifying relationships between entities
3. **Coreference Resolution**: Resolving entity mentions
4. **Entity Typing**: Fine-grained entity classification
5. **Event Extraction**: Identifying events involving entities

## Future Directions

### Research Areas
1. **Multilingual NER**: Cross-lingual entity recognition
2. **Domain Adaptation**: Specialized NER for specific domains
3. **Few-shot Learning**: NER with limited training data
4. **Real-time Processing**: Streaming NER applications

### Technical Improvements
1. **Better Tokenization**: Improved Korean tokenization methods
2. **Context Understanding**: Enhanced context-aware entity recognition
3. **Entity Linking**: Integration with knowledge bases
4. **Error Analysis**: Automated error pattern detection

## References

1. Park, S., et al. "KLUE: Korean Language Understanding Evaluation." arXiv preprint arXiv:2105.09680 (2021).
2. Lee, J., et al. "Korean Named Entity Recognition with Character-level Bidirectional LSTM-CNNs." arXiv preprint arXiv:1609.04913 (2016).
3. Kim, J., et al. "Korean Named Entity Recognition using BERT." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (2019).

## Additional Resources

- [KLUE Official Repository](https://github.com/KLUE-benchmark/KLUE)
- [Hugging Face KLUE Dataset](https://huggingface.co/datasets/klue)
- [Korean NLP Resources](https://github.com/ko-nlp/awesome-korean-nlp)
- [Korean Language Processing Papers](https://github.com/ko-nlp/papers) 