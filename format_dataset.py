import json
import re

def format_text(text):
    if not text:
        return ""
    # Remove common redundant headers in output
    text = text.replace("【回答内容】", "").strip()
    return text

def process_dataset(input_file, output_file):
    print(f"Reading from {input_file}...")
    
    formatted_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    
                    # 1. Cleaning Instruction
                    instruction = data.get('instruction', '').strip()
                    
                    # 2. Filling Input
                    # If input is empty, context is likely Hirakata City generally
                    input_text = data.get('input', '').strip()
                    if not input_text:
                        input_text = ""
                        
                    # 3. Cleaning Output
                    output_text = format_text(data.get('output', ''))
                    
                    # Reconstruct
                    new_entry = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": output_text
                    }
                    formatted_data.append(new_entry)
                    
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line[:50]}... Error: {e}")
                    
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print(f"Writing {len(formatted_data)} items to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print("Done.")

if __name__ == "__main__":
    process_dataset("FAQdataset.json", "FAQdataset2.json")
