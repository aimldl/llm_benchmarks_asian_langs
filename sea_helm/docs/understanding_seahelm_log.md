# Understanding the SEA-HELM Log

Each task undergoes:
- an Inference phase where the model generates outputs and
- an Evaluation phase where metrics are calculated against ground truth.

```
2025-07-16 19:13:07 | INFO | seahelm_evaluation | ---------- Inference | Lang: TL | Task: BELEBELE-QA-MC ----------
                                                | Testing Competency: NLU
```

## Lang (Language)
- TL: Tagalog
- ID: Indonesian
- VI: Vietnamese
- TH: Thai
- TA: Tamil

## Testing Competency
- NLU (Natural Language Understanding)
- SAFETY
- NLG (Natural Language Generation)
- NLR (Natural Language Reasoning)
- INSTRUCTION-FOLLOWING
- CULTURAL (specifically for Tagalog)
- LINGUISTIC-DIAGNOSTICS (Pragmatics, specifically Scalar Implicatures and Presuppositions for Indonesian and Tamil)
