#!/usr/bin/env python3
import json
from pathlib import Path

def debug_dst_extraction():
    """Debug the DST extraction to see what's happening with slot values."""
    
    # Read the JSON file
    json_path = Path('klue_dst/eval_dataset/klue-dst-v1.1_dev.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        # Read first few lines
        for i, line in enumerate(f):
            if i >= 3:  # Only process first 3 samples
                break
                
            line = line.strip()
            if line:
                sample = json.loads(line)
                
                print(f"\n=== Sample {i} ===")
                print(f"GUID: {sample.get('guid')}")
                print(f"Domains: {sample.get('domains')}")
                
                dialogue = sample.get('dialogue', [])
                print(f"Dialogue turns: {len(dialogue)}")
                
                # Check the last turn
                if dialogue:
                    last_turn = dialogue[-1]
                    print(f"Last turn role: {last_turn.get('role')}")
                    print(f"Last turn text: {last_turn.get('text', '')[:100]}...")
                    print(f"Last turn state: {last_turn.get('state', [])}")
                    
                    # Test slot parsing
                    state = last_turn.get('state', [])
                    slot_values = {}
                    requested_slots = []
                    
                    for slot in state:
                        print(f"Processing slot: {slot}")
                        if '-' in slot:
                            parts = slot.split('-', 2)
                            print(f"  Parts: {parts}")
                            if len(parts) >= 3:
                                domain = parts[0]
                                slot_name = parts[1]
                                value = parts[2]
                                print(f"  Domain: {domain}, Slot: {slot_name}, Value: {value}")
                                
                                if domain not in slot_values:
                                    slot_values[domain] = {}
                                slot_values[domain][slot_name] = value
                                requested_slots.append(f"{domain}-{slot_name}")
                    
                    print(f"Extracted slot_values: {slot_values}")
                    print(f"Extracted requested_slots: {requested_slots}")

if __name__ == "__main__":
    debug_dst_extraction() 