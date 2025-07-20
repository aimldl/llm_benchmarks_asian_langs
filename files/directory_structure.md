# Directory and File Structure

```bash
$ tree -d -L 3 llm_benchmarks_asian_langs
```
```bash
llm_benchmarks_asian_langs
├── __pycache__
├── files
│   ├── files
│   └── images
├── images
├── klue_dp
│   ├── benchmark_results
│   ├── eval_dataset
│   ├── logs
│   └── result_analysis
├── klue_dst
│   ├── __pycache__
│   ├── benchmark_results
│   ├── eval_dataset
│   ├── logs
│   └── result_analysis
├── klue_mrc
│   ├── __pycache__
│   ├── benchmark_results
│   ├── eval_dataset
│   ├── logs
│   └── result_analysis
├── klue_ner
│   ├── __pycache__
│   ├── benchmark_results
│   ├── eval_dataset
│   ├── logs
│   └── result_analysis
├── klue_nli
│   ├── benchmark_results
│   ├── eval_dataset
│   ├── logs
│   └── result_analysis
├── klue_re
│   ├── __pycache__
│   ├── benchmark_results
│   ├── eval_dataset
│   ├── logs
│   ├── result_analysis
│   └── venv
│       ├── bin
│       ├── include
│       ├── lib
│       └── lib64 -> lib
├── klue_sts
│   └── eval_dataset
└── klue_tc
    ├── __pycache__
    ├── benchmark_results
    ├── eval_dataset
    ├── logs
    └── result_analysis

53 directories
$
```

## TODO
Clean up things like
```bash
├── klue_re
   ...
│   └── venv
│       ├── bin
│       ├── include
│       ├── lib
│       └── lib64 -> lib
```
Make sure to verify the code and other stuff before clearning it up.

