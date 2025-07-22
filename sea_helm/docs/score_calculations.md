# Score calculations
## Normalization process
Score normalization is done to account for different difficulties and random baseline scores for each task.

The calculation of the normalized scores is:

```math
\text{normalised\_score} = (\text{score} - \text{baseline}) / (\text{maximum} - \text{baseline}) * 100
```
where $\text{baseline}$ is equal to:
* Multiple choice tasks: $`1/\text{n\_options}`$
* Generative tasks: $0$

and $\text{maximum}$ is equal to:
* maximum possible score that an answer can get (typically $1$)

## Aggregation process
Each task in SEA-HELM is grouped into one of the following competencies - NLU, NLG, NLR, Instruction-Following, Multi-Turn, Cultural, Safety.

The following aggregated scores are calculated:
1. Competency score
    - The average of all the normalised metrics from each task
    - Example:
        > Competency: NLR  
        > Tasks to average: NLI, Causal
2. Language score
    - The average score of all the competencies for the each language
    - Example:
        > Language: ID  
        > Competencies to average: NLU, NLG, NLR, Instruction-Following, Multi-Turn, Safety
3. SEA Average
    - The average of all the language scores
    - Example:
        > SEA Average  
        > Languages to average: FIL. ID, TA, TH, VI

> [!Note]  
> **Sub-task aggregation**   
> For tasks with sub-task (e.g translation - translation-xx-en and translation-en-xx), the scores for each sub-tasks are first averaged to get the task score. This task score is then used in the calculation of the competency score.
