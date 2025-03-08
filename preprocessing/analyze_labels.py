import json
from collections import defaultdict
from IPython.display import display, HTML
import pandas as pd

def analyze_annotations(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize counters and storage
    entity_counts = defaultdict(int)
    relation_counts = defaultdict(int)
    entity_examples = defaultdict(list)
    relation_examples = defaultdict(list)
    triple_examples = defaultdict(list)
    
    # Analyze annotations
    for item in data:
        if 'annotations' in item:
            # Store entities for this text segment
            entities_in_text = {}
            
            # First pass: collect entities
            for idx, ann in enumerate(item['annotations']):
                if ann['type'] == 'entity':
                    label = ann['label']
                    text = ann['text']
                    entity_counts[label] += 1
                    entities_in_text[idx] = {'text': text, 'label': label}
                    
                    # Store unique examples (up to 3)
                    if len(entity_examples[label]) < 3 and text not in entity_examples[label]:
                        entity_examples[label].append(text)

            # Second pass: collect relations and triples
            for ann in item['annotations']:
                if ann['type'] == 'relation':
                    relation_type = ann['label']
                    relation_counts[relation_type] += 1
                    
                    # Get the related entities
                    from_entity = entities_in_text.get(ann['from_entity'])
                    to_entity = entities_in_text.get(ann['to_entity'])
                    
                    if from_entity and to_entity:
                        triple = {
                            'from': from_entity,
                            'relation': relation_type,
                            'to': to_entity,
                            'context': ann.get('text', '')
                        }
                        
                        # Store unique triple examples (up to 3)
                        if len(triple_examples[relation_type]) < 3:
                            triple_examples[relation_type].append(triple)

    # Print summaries
    print("\n=== Entity Types Summary ===")
    print(f"{'Entity Type':<20} {'Count':<8} {'Example Texts':<40}")
    print("-" * 68)
    for entity_type, count in sorted(entity_counts.items()):
        examples = ', '.join(entity_examples[entity_type])
        print(f"{entity_type:<20} {count:<8} {examples:<40}")

    print("\n=== Relation Types Summary ===")
    print(f"{'Relation Type':<20} {'Count':<8}")
    print("-" * 28)
    for relation_type, count in sorted(relation_counts.items()):
        print(f"\n{relation_type:<20} {count:<8}")
        print("Example Triples:")
        for triple in triple_examples[relation_type]:
            print(f"    {triple['from']['text']} ({triple['from']['label']}) "
                  f"--[{triple['relation']}]--> "
                  f"{triple['to']['text']} ({triple['to']['label']})")
            if triple['context']:
                print(f"    Context: '{triple['context']}'")

    return {
        'entity_counts': dict(entity_counts),
        'relation_counts': dict(relation_counts),
        'triple_examples': dict(triple_examples)
    }

if __name__ == "__main__":
    json_file = r"./data/output/annotated_text.json"
    analyze_annotations(json_file)