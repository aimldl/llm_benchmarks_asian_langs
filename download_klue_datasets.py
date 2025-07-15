#!/usr/bin/env python3
"""
KLUE Dataset Downloader and Extractor

This script downloads all KLUE benchmark datasets and extracts the significant values
into CSV files for easy review and analysis.

Tasks covered:
- DP (Dependency Parsing)
- DST (Dialogue State Tracking) 
- MRC (Machine Reading Comprehension)
- NER (Named Entity Recognition)
- NLI (Natural Language Inference)
- RE (Relation Extraction)
- STS (Semantic Textual Similarity)
- TC (Topic Classification) - already exists
"""

import os
import json
import csv
import requests
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Base URL for KLUE datasets
KLUE_BASE_URL = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark"

# Dataset configurations
DATASET_CONFIGS = {
    'klue_dp': {
        'url': f"{KLUE_BASE_URL}/klue-dp-v1.1/klue-dp-v1.1_dev.tsv",
        'filename': 'klue-dp-v1.1_dev.tsv',
        'type': 'tsv'
    },
    'klue_mrc': {
        'url': f"{KLUE_BASE_URL}/klue-mrc-v1.1/klue-mrc-v1.1_dev.json",
        'filename': 'klue-mrc-v1.1_dev.json',
        'type': 'json'
    },
    'klue_ner': {
        'url': f"{KLUE_BASE_URL}/klue-ner-v1.1/klue-ner-v1.1_dev.tsv",
        'filename': 'klue-ner-v1.1_dev.tsv',
        'type': 'tsv'
    },
    'klue_nli': {
        'url': f"{KLUE_BASE_URL}/klue-nli-v1.1/klue-nli-v1.1_dev.json",
        'filename': 'klue-nli-v1.1_dev.json',
        'type': 'json'
    },
    'klue_re': {
        'url': f"{KLUE_BASE_URL}/klue-re-v1.1/klue-re-v1.1_dev.json",
        'filename': 'klue-re-v1.1_dev.json',
        'type': 'json'
    },
    'klue_sts': {
        'url': f"{KLUE_BASE_URL}/klue-sts-v1.1/klue-sts-v1.1_dev.json",
        'filename': 'klue-sts-v1.1_dev.json',
        'type': 'json'
    }
}

def download_file(url, filepath):
    """Download a file from URL to the specified filepath."""
    print(f"Downloading {url} to {filepath}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Downloaded {filepath}")

def download_dst_dataset(task_dir):
    """Download DST dataset from Hugging Face."""
    print("Downloading KLUE DST (Wizard of Seoul) dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset('klue', 'wos', split='validation')
        
        # Save as JSON
        json_path = task_dir / 'eval_dataset' / 'klue-dst-v1.1_dev.json'
        dataset.to_json(str(json_path))
        print(f"✓ Downloaded DST dataset to {json_path}")
        
        # Extract significant values to CSV
        extract_dst_to_csv(dataset, task_dir)
        
    except Exception as e:
        print(f"✗ Failed to download DST dataset: {e}")
        # Try to extract from existing JSON file if download failed
        json_path = task_dir / 'eval_dataset' / 'klue-dst-v1.1_dev.json'
        if json_path.exists():
            print("Attempting to extract from existing JSON file...")
            try:
                # Read JSONL format
                dataset = []
                with open(json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            dataset.append(json.loads(line))
                
                # Extract significant values to CSV
                extract_dst_to_csv(dataset, task_dir)
                return True
            except Exception as e2:
                print(f"✗ Failed to extract from existing JSON file: {e2}")
                return False
        return False
    
    return True

def extract_dst_to_csv(dataset, task_dir):
    """Extract significant values from DST dataset to CSV."""
    csv_path = task_dir / 'eval_dataset' / 'klue-dst-v1.1_dev_extracted.csv'
    
    print(f"Extracting DST data to {csv_path}...")
    
    extracted_data = []
    for i, sample in enumerate(dataset):
        # Extract key information
        guid = sample.get('guid', f'dialogue_{i}')
        domains = sample.get('domains', [])
        dialogue = sample.get('dialogue', [])
        
        # Get dialogue context (last few turns)
        context = ""
        if dialogue:
            # Take last 3 turns for context
            recent_turns = dialogue[-3:] if len(dialogue) > 3 else dialogue
            context_parts = []
            for turn in recent_turns:
                role = turn.get('role', '')
                text = turn.get('text', '')
                # Truncate long text
                if len(text) > 100:
                    text = text[:100] + '...'
                context_parts.append(f"{role}: {text}")
            context = " | ".join(context_parts)
        
        # Extract slot values from the last user turn's state
        slot_values = {}
        if dialogue:
            # Find the last user turn with state
            last_user_turn = None
            for turn in reversed(dialogue):
                if turn.get('role') == 'user' and turn.get('state'):
                    last_user_turn = turn
                    break
            
            if last_user_turn:
                state = last_user_turn.get('state', [])
                for slot in state:
                    if '-' in slot:
                        parts = slot.split('-', 2)  # Split into max 3 parts
                        if len(parts) >= 3:
                            domain = parts[0]
                            slot_name = parts[1]
                            value = parts[2]
                            if domain not in slot_values:
                                slot_values[domain] = {}
                            slot_values[domain][slot_name] = value
        
        # Count dialogue turns
        turn_count = len(dialogue)
        
        # Get active intent (usually from the last user turn)
        active_intent = ""
        for turn in reversed(dialogue):
            if turn.get('role') == 'user':
                # Extract intent from the text (simplified)
                text = turn.get('text', '')
                if any(keyword in text for keyword in ['예약', '찾', '알려', '문의']):
                    active_intent = "information_request"
                elif any(keyword in text for keyword in ['예약', '부탁', '해주']):
                    active_intent = "booking_request"
                else:
                    active_intent = "general_inquiry"
                break
        
        # Get requested slots (slots that have values in the last user turn)
        requested_slots = []
        if dialogue:
            # Find the last user turn with state
            last_user_turn = None
            for turn in reversed(dialogue):
                if turn.get('role') == 'user' and turn.get('state'):
                    last_user_turn = turn
                    break
            
            if last_user_turn:
                state = last_user_turn.get('state', [])
                for slot in state:
                    if '-' in slot:
                        parts = slot.split('-', 2)
                        if len(parts) >= 3:
                            domain = parts[0]
                            slot_name = parts[1]
                            requested_slots.append(f"{domain}-{slot_name}")
        
        extracted_data.append({
            'id': i,
            'dialogue_id': guid,
            'turn_id': turn_count,
            'domains': ';'.join(domains),
            'active_intent': active_intent,
            'requested_slots': ';'.join(requested_slots),
            'slot_values': str(slot_values),
            'context': context
        })
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if extracted_data:
            writer = csv.DictWriter(f, fieldnames=extracted_data[0].keys())
            writer.writeheader()
            writer.writerows(extracted_data)
    
    print(f"✓ Extracted {len(extracted_data)} DST samples to CSV")

def extract_dp_to_csv(tsv_path, task_dir):
    """Extract significant values from DP dataset to CSV."""
    csv_path = task_dir / 'eval_dataset' / 'klue-dp-v1.1_dev_extracted.csv'
    
    print(f"Extracting DP data to {csv_path}...")
    
    extracted_data = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            # Parse TSV format: guid, sentence, words, pos_tags, heads, deprels
            parts = line.split('\t')
            if len(parts) >= 6:
                guid = parts[0]
                sentence = parts[1]
                words = parts[2].split(' ')
                pos_tags = parts[3].split(' ')
                heads = parts[4].split(' ')
                deprels = parts[5].split(' ')
                
                # Count different POS tags and dependency relations
                unique_pos = len(set(pos_tags))
                unique_deprels = len(set(deprels))
                word_count = len(words)
                
                extracted_data.append({
                    'id': i,
                    'guid': guid,
                    'sentence': sentence,
                    'word_count': word_count,
                    'unique_pos_tags': unique_pos,
                    'unique_deprels': unique_deprels,
                    'words': ' '.join(words[:10]) + ('...' if len(words) > 10 else ''),  # First 10 words
                    'pos_tags': ' '.join(pos_tags[:10]) + ('...' if len(pos_tags) > 10 else ''),
                    'deprels': ' '.join(deprels[:10]) + ('...' if len(deprels) > 10 else '')
                })
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if extracted_data:
            writer = csv.DictWriter(f, fieldnames=extracted_data[0].keys())
            writer.writeheader()
            writer.writerows(extracted_data)
    
    print(f"✓ Extracted {len(extracted_data)} DP samples to CSV")

def extract_ner_to_csv(tsv_path, task_dir):
    """Extract significant values from NER dataset to CSV."""
    csv_path = task_dir / 'eval_dataset' / 'klue-ner-v1.1_dev_extracted.csv'
    
    print(f"Extracting NER data to {csv_path}...")
    
    extracted_data = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        sentence_id = 0
        
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    # Process completed sentence
                    words = [item[0] for item in current_sentence]
                    tags = [item[1] for item in current_sentence]
                    
                    # Count entities
                    entity_count = sum(1 for tag in tags if tag.startswith('B-'))
                    unique_entity_types = len(set([tag[2:] for tag in tags if tag.startswith('B-')]))
                    
                    extracted_data.append({
                        'id': sentence_id,
                        'sentence': ' '.join(words),
                        'word_count': len(words),
                        'entity_count': entity_count,
                        'unique_entity_types': unique_entity_types,
                        'entity_types': ';'.join(set([tag[2:] for tag in tags if tag.startswith('B-')])),
                        'words': ' '.join(words[:15]) + ('...' if len(words) > 15 else ''),
                        'tags': ' '.join(tags[:15]) + ('...' if len(tags) > 15 else '')
                    })
                    
                    current_sentence = []
                    sentence_id += 1
            else:
                # Parse word and tag
                parts = line.split('\t')
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[1]
                    current_sentence.append((word, tag))
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if extracted_data:
            writer = csv.DictWriter(f, fieldnames=extracted_data[0].keys())
            writer.writeheader()
            writer.writerows(extracted_data)
    
    print(f"✓ Extracted {len(extracted_data)} NER samples to CSV")

def extract_json_to_csv(json_path, task_dir, task_type):
    """Extract significant values from JSON datasets to CSV."""
    filename = json_path.name.replace('.json', '_extracted.csv')
    csv_path = task_dir / 'eval_dataset' / filename
    
    print(f"Extracting {task_type} data to {csv_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Handle different JSON structures
    if task_type == 'mrc':
        # MRC has a different structure with 'data' field
        data = raw_data.get('data', [])
        # Flatten the data structure
        flattened_data = []
        for article in data:
            title = article.get('title', '')
            paragraphs = article.get('paragraphs', [])
            for paragraph in paragraphs:
                context = paragraph.get('context', '')
                qas = paragraph.get('qas', [])
                for qa in qas:
                    qa['title'] = title
                    qa['context'] = context
                    flattened_data.append(qa)
        data = flattened_data
    else:
        data = raw_data
    
    extracted_data = []
    
    for i, sample in enumerate(data):
        if task_type == 'mrc':
            # MRC: Machine Reading Comprehension
            guid = sample.get('guid', f'mrc_{i}')
            title = sample.get('title', '')
            context = sample.get('context', '')
            question = sample.get('question', '')
            answers = sample.get('answers', {})
            # answers can be a dict (with 'text' key) or a list (SQuAD-style)
            answer_text = ''
            if isinstance(answers, dict):
                answer_text = answers.get('text', [''])[0] if answers.get('text') else ''
            elif isinstance(answers, list) and answers:
                # Sometimes answers is a list of dicts
                first = answers[0]
                if isinstance(first, dict):
                    answer_text = first.get('text', '')
                else:
                    answer_text = str(first)
            
            extracted_data.append({
                'id': i,
                'guid': guid,
                'title': title[:100] + '...' if len(title) > 100 else title,
                'context_length': len(context),
                'question': question[:100] + '...' if len(question) > 100 else question,
                'answer': answer_text[:100] + '...' if len(answer_text) > 100 else answer_text,
                'context_preview': context[:200] + '...' if len(context) > 200 else context
            })
            
        elif task_type == 'nli':
            # NLI: Natural Language Inference
            guid = sample.get('guid', f'nli_{i}')
            premise = sample.get('premise', '')
            hypothesis = sample.get('hypothesis', '')
            label = sample.get('label', '')
            
            extracted_data.append({
                'id': i,
                'guid': guid,
                'premise': premise[:100] + '...' if len(premise) > 100 else premise,
                'hypothesis': hypothesis[:100] + '...' if len(hypothesis) > 100 else hypothesis,
                'label': label,
                'premise_length': len(premise),
                'hypothesis_length': len(hypothesis)
            })
            
        elif task_type == 're':
            # RE: Relation Extraction
            guid = sample.get('guid', f're_{i}')
            sentence = sample.get('sentence', '')
            subject_entity = sample.get('subject_entity', {})
            object_entity = sample.get('object_entity', {})
            label = sample.get('label', '')
            
            subject_text = subject_entity.get('word', '') if subject_entity else ''
            object_text = object_entity.get('word', '') if object_entity else ''
            
            extracted_data.append({
                'id': i,
                'guid': guid,
                'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                'subject_entity': subject_text,
                'object_entity': object_text,
                'relation': label,
                'sentence_length': len(sentence)
            })
            
        elif task_type == 'sts':
            # STS: Semantic Textual Similarity
            guid = sample.get('guid', f'sts_{i}')
            sentence1 = sample.get('sentence1', '')
            sentence2 = sample.get('sentence2', '')
            score = sample.get('score', 0)
            
            extracted_data.append({
                'id': i,
                'guid': guid,
                'sentence1': sentence1[:100] + '...' if len(sentence1) > 100 else sentence1,
                'sentence2': sentence2[:100] + '...' if len(sentence2) > 100 else sentence2,
                'similarity_score': score,
                'sentence1_length': len(sentence1),
                'sentence2_length': len(sentence2)
            })
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if extracted_data:
            writer = csv.DictWriter(f, fieldnames=extracted_data[0].keys())
            writer.writeheader()
            writer.writerows(extracted_data)
    
    print(f"✓ Extracted {len(extracted_data)} {task_type.upper()} samples to CSV")

def main():
    """Main function to download and extract all KLUE datasets."""
    print("=" * 60)
    print("KLUE Dataset Downloader and Extractor")
    print("=" * 60)
    
    # Get the current directory (should be the root of the project)
    current_dir = Path.cwd()
    
    # Process each task
    for task_name, config in DATASET_CONFIGS.items():
        print(f"\n{'='*20} Processing {task_name.upper()} {'='*20}")
        
        task_dir = current_dir / task_name
        eval_dataset_dir = task_dir / 'eval_dataset'
        
        # Create eval_dataset directory if it doesn't exist
        eval_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists
        dataset_path = eval_dataset_dir / config['filename']
        if dataset_path.exists():
            print(f"✓ Dataset already exists: {dataset_path}")
        else:
            try:
                # Download the dataset
                download_file(config['url'], dataset_path)
            except Exception as e:
                print(f"✗ Failed to download {task_name}: {e}")
                continue
        
        # Extract to CSV based on file type
        if config['type'] == 'tsv':
            if task_name == 'klue_dp':
                extract_dp_to_csv(dataset_path, task_dir)
            elif task_name == 'klue_ner':
                extract_ner_to_csv(dataset_path, task_dir)
        elif config['type'] == 'json':
            task_type = task_name.split('_')[1]  # Extract task type (mrc, nli, re, sts)
            extract_json_to_csv(dataset_path, task_dir, task_type)
    
    # Handle DST separately (uses Hugging Face)
    print(f"\n{'='*20} Processing KLUE_DST {'='*20}")
    dst_task_dir = current_dir / 'klue_dst'
    if dst_task_dir.exists():
        download_dst_dataset(dst_task_dir)
    else:
        print("✗ klue_dst directory not found")
    
    # TC is already handled (exists in klue_tc/eval_dataset)
    print(f"\n{'='*20} Processing KLUE_TC {'='*20}")
    tc_task_dir = current_dir / 'klue_tc'
    tc_dataset_path = tc_task_dir / 'eval_dataset' / 'ynat-v1.1_dev.json'
    tc_csv_path = tc_task_dir / 'eval_dataset' / 'ynat-v1.1_dev_extracted.csv'
    
    if tc_dataset_path.exists():
        print(f"✓ TC dataset already exists: {tc_dataset_path}")
        if tc_csv_path.exists():
            print(f"✓ TC extracted CSV already exists: {tc_csv_path}")
        else:
            print("✗ TC extracted CSV missing - you may want to regenerate it")
    else:
        print("✗ TC dataset not found")
    
    print(f"\n{'='*60}")
    print("Dataset download and extraction completed!")
    print("=" * 60)
    print("\nSummary of downloaded datasets:")
    
    # List all downloaded datasets
    for task_name in ['klue_dp', 'klue_dst', 'klue_mrc', 'klue_ner', 'klue_nli', 'klue_re', 'klue_sts', 'klue_tc']:
        task_dir = current_dir / task_name
        eval_dataset_dir = task_dir / 'eval_dataset'
        
        if eval_dataset_dir.exists():
            files = list(eval_dataset_dir.glob('*'))
            if files:
                print(f"✓ {task_name}: {len(files)} files")
                for file in files:
                    print(f"  - {file.name}")
            else:
                print(f"✗ {task_name}: No files found")
        else:
            print(f"✗ {task_name}: eval_dataset directory not found")

if __name__ == "__main__":
    main() 