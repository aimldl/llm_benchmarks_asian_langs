# Performance Metrics for Machine Reading Comprehension

The evaluation of machine reading comprehension (MRC) models often employs **Exact Match (EM)** and **ROUGE-W** to ensure a comprehensive assessment of performance. This combination balances the need for precise reproduction (EM) with a degree of flexibility for minor variations in generated text (ROUGE-W).

### Exact Match (EM)

**Exact Match (EM)** is a stringent performance metric. An answer is considered correct only if it is an **exact, character-for-character match** with the reference answer. This metric is useful for evaluating the model's ability to precisely identify and reproduce the correct text span.

### ROUGE-W (Weighted Longest Common Subsequence-based)
This section details the ROUGE metric family, including ROUGE-N (e.g., ROUGE-1, ROUGE-2), ROUGE-L, and ROUGE-W. The explanation of ROUGE-W's relationship with LCCS-based F1 is also provided.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** is a set of metrics primarily utilized for assessing automatic summarization and machine translation systems. ROUGE evaluates the quality of a machine-generated text (candidate) by comparing it against a human-written reference text (gold standard), with higher scores indicating greater similarity. The focus of ROUGE is primarily on **recall**, measuring the extent to which information from the reference text is captured by the candidate text.

ROUGE metrics typically report scores for **Precision**, **Recall**, and **F1-score**.

* **Precision:** Quantifies the proportion of the generated text that is relevant to the reference.
    $$\text{Precision} = \frac{\text{Number of overlapping units}}{\text{Number of units in generated text}}$$
* **Recall:** Measures the proportion of the reference text covered by the generated text.
    $$\text{Recall} = \frac{\text{Number of overlapping units}}{\text{Number of units in reference text}}$$
* **F1-score:** Represents the harmonic mean of precision and recall, providing a balanced measure.
    $$\text{F1-score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Types of ROUGE Metrics

* **ROUGE-N (ROUGE-1, ROUGE-2, etc.):** This variant measures the overlap of **n-grams** (contiguous sequences of $n$ words) between the candidate and reference texts.
    * **ROUGE-1:** Focuses on **unigrams** (individual words), assessing basic content overlap.
    * **ROUGE-2:** Focuses on **bigrams** (pairs of consecutive words), offering insights into the fluency and coherence of phrases.
    Higher ROUGE-N scores generally indicate that the generated text contains more of the same words and short phrases as the reference.

* **ROUGE-L (Longest Common Subsequence-based):** Unlike ROUGE-N, ROUGE-L utilizes the **Longest Common Subsequence (LCS)**. The LCS is the longest sequence of words that appears in both texts in the same order, though not necessarily consecutively. ROUGE-L is more flexible as it does not demand contiguous matches, thereby capturing structural similarity and sentence-level coherence even with word reordering or omissions. This is particularly useful when exact phrasing is less critical than the overall flow of information.

* **ROUGE-W (Weighted Longest Common Subsequence-based):** An extension of ROUGE-L, ROUGE-W assigns **higher weights to consecutive matches** compared to non-consecutive matches. This means that extensive, exact sequences of words shared between the candidate and reference texts receive significantly higher scores. While still considering the LCS, ROUGE-W prioritizes longer, unbroken common sequences.

#### LCCS-based F1 and ROUGE-W

The term "LCCS-based F1" directly pertains to ROUGE-W. ROUGE-W is fundamentally constructed upon the concept of the **Longest Common Consecutive Subsequence (LCCS)**. While ROUGE-L permits gaps within the common subsequence, ROUGE-W's weighting scheme explicitly emphasizes consecutive matches. Therefore, when "LCCS-based F1" is specified as a performance metric for tasks such as KLUE-MRC, it signifies the use of the **ROUGE-W F1-score**. This emphasis highlights the metric's prioritization of accurate and unbroken sequences of words, which is crucial for extractive question answering.


#### Rationale for ROUGE-W in KLUE-MRC

The **KLUE-MRC (Machine Reading Comprehension)** task typically involves extractive question answering, where the model's objective is to extract a precise span of text from a given passage to answer a question. For such tasks, the following aspects are critical:

* **Exact wording and order:** For an extracted answer to be correct, the words must be present in the correct order to form a coherent and accurate span.
* **Long, precise matches:** A longer, perfectly matching span is considerably more desirable than several scattered matching words.
* **Penalty for extraneous words:** While recall (capturing all necessary words) is important, precision also plays a vital role in preventing the inclusion of irrelevant words within the extracted answer.

ROUGE-W is particularly well-suited for KLUE-MRC for the following reasons:

* **Beyond simple word overlap (ROUGE-1):** ROUGE-1 only indicates the presence of individual words and cannot differentiate between syntactically correct and nonsensical phrases.
* **Greater flexibility than strict n-grams (ROUGE-2):** While ROUGE-2 considers word pairs, it can be overly rigid. Minor reordering or omission of a single word in a longer sequence can heavily penalize ROUGE-2, even if the overall meaning and span remain largely correct.
* **Prioritization of contiguous sequences (ROUGE-L vs. ROUGE-W):** ROUGE-L considers the longest common subsequence and tolerates gaps. However, for MRC, a perfect or near-perfect consecutive match is often required. ROUGE-W's weighting scheme directly addresses this by awarding more credit to longer, unbroken common segments, which is characteristic of high-quality extractive answers. Essentially, it identifies and heavily weights the "longest common consecutive subsequence."

In conclusion, the **ROUGE-W's LCCS-based F1 score** achieves a crucial balance. It offers sufficient flexibility to account for minor variations (unlike the strictness of EM, which demands an exact match) while heavily rewarding the consecutive, accurate text spans that define effective answers in extractive MRC. This makes ROUGE-W a robust metric for evaluating a model's proficiency in pinpointing and reproducing precise answers from source texts.