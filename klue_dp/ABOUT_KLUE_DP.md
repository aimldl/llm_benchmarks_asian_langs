# About KLUE Dependency Parsing

The KLUE Dependency Parsing task is one of 8 Korean natural language understanding (NLU) tasks in the Korean Language Understanding Evaluation (KLUE) benchmark. The KLUE benchmark serves as a standard for evaluating a model's ability to comprehend and analyze Korean text at the syntactic level. This task specifically focuses on analyzing the grammatical structure of Korean sentences by identifying dependency relationships between words.

## Dataset: Korean Dependency Parsing

The dataset used for this task is the **Korean Dependency Parsing** dataset, which contains Korean sentences annotated with part-of-speech tags and dependency relations. The dataset is designed to evaluate a model's ability to understand Korean syntax and grammatical structure.

### Task Description

Dependency parsing involves two main components:

1. **Part-of-Speech (POS) Tagging**: Identifying the grammatical category of each word in a sentence
2. **Dependency Analysis**: Determining which word each word depends on (head) and the type of dependency relationship

### Korean POS Tags

The dataset uses 35 Korean POS tags following the Korean Treebank (KTB) standard:

#### Content Words
- **NNG**: 일반명사 (Common Noun)
- **NNP**: 고유명사 (Proper Noun)
- **NNB**: 의존명사 (Dependent Noun)
- **NNBC**: 단위를 나타내는 명사 (Unit Noun)
- **NR**: 수사 (Numeral)
- **NP**: 대명사 (Pronoun)
- **VV**: 동사 (Verb)
- **VA**: 형용사 (Adjective)
- **VX**: 보조용언 (Auxiliary Verb)
- **VCP**: 긍정지정사 (Positive Copula)
- **VCN**: 부정지정사 (Negative Copula)
- **MM**: 관형사 (Determiner)
- **MAG**: 일반부사 (General Adverb)
- **MAJ**: 접속부사 (Conjunctive Adverb)
- **IC**: 감탄사 (Interjection)

#### Particles
- **JKS**: 주격조사 (Subject Particle)
- **JKC**: 보격조사 (Complement Particle)
- **JKG**: 관형격조사 (Adnominal Particle)
- **JKO**: 목적격조사 (Object Particle)
- **JKB**: 부사격조사 (Adverbial Particle)
- **JKV**: 호격조사 (Vocative Particle)
- **JKQ**: 인용격조사 (Quotative Particle)
- **JX**: 보조사 (Auxiliary Particle)
- **JC**: 접속조사 (Conjunctive Particle)

#### Endings
- **EP**: 선어말어미 (Pre-final Ending)
- **EF**: 종결어미 (Final Ending)
- **EC**: 연결어미 (Connective Ending)
- **ETN**: 명사형전성어미 (Nominal Ending)
- **ETM**: 관형형전성어미 (Adnominal Ending)

#### Others
- **XPN**: 체언접두사 (Noun Prefix)
- **XSN**: 명사파생접미사 (Noun Derivational Suffix)
- **XSV**: 동사파생접미사 (Verb Derivational Suffix)
- **XSA**: 형용사파생접미사 (Adjective Derivational Suffix)
- **XR**: 어근 (Root)
- **SF**: 마침표,물음표,느낌표 (Sentence-final Punctuation)
- **SP**: 쉼표,가운뎃점,콜론,빗금 (Comma, Middle Dot, Colon, Slash)
- **SS**: 따옴표,괄호표,줄표 (Quotation Marks, Brackets, Dash)
- **SE**: 줄임표 (Ellipsis)
- **SO**: 붙임표(물결,숨김,빠짐) (Attachment Mark)
- **SW**: 기타기호(논리수학기호,화폐기호) (Other Symbols)
- **SL**: 외국어 (Foreign Language)
- **SH**: 한자 (Chinese Character)
- **SN**: 숫자 (Number)

### Dependency Relations

The dataset includes various dependency relations that describe the grammatical relationships between words:

#### Core Arguments
- **nsubj**: 주어 (Subject)
- **obj**: 목적어 (Object)
- **iobj**: 간접목적어 (Indirect Object)
- **ccomp**: 보문 (Clausal Complement)
- **xcomp**: 개방형 보문 (Open Clausal Complement)

#### Modifiers
- **amod**: 형용사 수식어 (Adjectival Modifier)
- **nummod**: 수사 수식어 (Numeric Modifier)
- **det**: 한정사 (Determiner)
- **advmod**: 부사 수식어 (Adverbial Modifier)
- **advcl**: 부사절 (Adverbial Clause)

#### Function Words
- **case**: 격조사 (Case Marker)
- **mark**: 접속조사 (Subordinating Conjunction)
- **aux**: 보조동사 (Auxiliary Verb)
- **cop**: 연결동사 (Copula)

#### Special Relations
- **root**: 루트 (Root)
- **punct**: 구두점 (Punctuation)
- **compound**: 복합어 (Compound)
- **flat**: 평면 구조 (Flat Structure)
- **list**: 나열 (List)
- **parataxis**: 병렬 구조 (Parataxis)
- **discourse**: 담화 표지 (Discourse Marker)
- **vocative**: 호격 (Vocative)
- **expl**: 가주어/가목적어 (Expletive)
- **acl**: 관계절 (Relative Clause)
- **appos**: 동격어 (Apposition)
- **dislocated**: 도치된 요소 (Dislocated Element)
- **orphan**: 고아 요소 (Orphan)
- **goeswith**: 연결된 요소 (Goes With)
- **reparandum**: 수정된 요소 (Reparandum)
- **dep**: 기타 의존 관계 (Other Dependency)

### Evaluation Metrics

The task is evaluated using two primary metrics:

1. **UAS (Unlabeled Attachment Score)**: The percentage of words that have their head correctly identified, regardless of the dependency label.

2. **LAS (Labeled Attachment Score)**: The percentage of words that have both their head and dependency label correctly identified.

### Dataset Structure

Each sample in the dataset contains:
- **guid**: Unique identifier for the sample
- **sentence**: The complete Korean sentence
- **words**: List of words in the sentence
- **pos_tags**: List of POS tags corresponding to each word
- **heads**: List of head indices (0-based) for each word
- **deprels**: List of dependency relation labels for each word

### Example

```json
{
    "guid": "klue-dp-v1.1_dev_00000",
    "sentence": "한국어 문장의 의존 구문 분석을 수행합니다.",
    "words": ["한국어", "문장", "의", "의존", "구문", "분석", "을", "수행", "합니다", "."],
    "pos_tags": ["NNG", "NNG", "JKG", "NNG", "NNG", "NNG", "JKO", "NNG", "EF", "SF"],
    "heads": [2, 6, 6, 6, 6, 8, 8, 9, 0, 9],
    "deprels": ["compound", "compound", "case", "compound", "compound", "obj", "case", "nsubj", "root", "punct"]
}
```

### Korean Language Characteristics

Korean dependency parsing presents unique challenges due to the language's agglutinative nature:

1. **Agglutination**: Korean words can have multiple morphemes attached, making word segmentation and dependency analysis complex.

2. **Particle-rich**: Korean uses many particles (조사) that indicate grammatical relationships, which must be properly analyzed.

3. **Word Order Flexibility**: While Korean generally follows SOV (Subject-Object-Verb) order, word order can be flexible, especially in spoken language.

4. **Honorific System**: Korean has a complex honorific system that affects verb endings and particle usage.

5. **Ellipsis**: Korean frequently omits subjects and objects when they can be inferred from context.

### Performance Expectations

Typical performance ranges for modern models on Korean dependency parsing:

- **UAS**: 75-90% (depending on model size and training data)
- **LAS**: 70-85% (dependency labels are more challenging than head identification)
- **Best Performance**: On simple sentences with clear dependency structures
- **Challenging Cases**: Complex sentences with multiple clauses, ambiguous dependencies, and ellipsis

### Applications

Korean dependency parsing has various applications:

1. **Machine Translation**: Understanding sentence structure improves translation quality
2. **Information Extraction**: Identifying relationships between entities
3. **Question Answering**: Understanding question structure and finding relevant answers
4. **Text Summarization**: Identifying key relationships for summarization
5. **Grammar Checking**: Detecting grammatical errors in Korean text
6. **Language Learning**: Helping learners understand Korean sentence structure

### References

For more information about the KLUE benchmark and dependency parsing:

- KLUE Paper: [https://arxiv.org/pdf/2105.09680](https://arxiv.org/pdf/2105.09680)
- KLUE GitHub: [https://github.com/KLUE-benchmark/KLUE](https://github.com/KLUE-benchmark/KLUE)
- Korean Treebank: [https://korean-treebank.readthedocs.io/](https://korean-treebank.readthedocs.io/) 