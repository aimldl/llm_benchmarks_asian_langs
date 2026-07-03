import datasets

#filepath='/home/user/benchmarking-sea-helm-with-gemini/SEA-HELM/seahelm_tasks/multi_turn/mt_bench/data/id_sea_mt_bench.jsonl'
dir_jsonl_error='/home/user/benchmarking-sea-helm-with-gemini/SEA-HELM/archive/2025-08-24-sun/jsonl_error/seahelm_tasks/multi_turn/mt_bench/data/solve_jsonl_formatting_issue'
#file_name = 'id_sea_mt_bench-local-contaminated.jsonl'
file_name = 'id_sea_mt_bench.jsonl'
filepath = dir_jsonl_error +'/'+file_name
print(filepath)

#dataset = datasets.load_dataset("json", data_files=[filepath], split="train")
dataset = datasets.load_dataset("json", data_files=filepath, split="train")
#print("dataset is loaded!")