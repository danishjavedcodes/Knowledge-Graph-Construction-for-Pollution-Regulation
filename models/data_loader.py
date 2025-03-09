import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json

class MaritimeRegulationsDataset(Dataset):
    def __init__(self, json_file, tokenizer_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Create label mappings
        self.entity_types = {
            'SYSTEM': 0, 'EQUIPMENT': 1, 'SPECIFICATION': 2,
            'DISPOSAL': 3, 'ACTIVITIES': 4, 'PENALTIES': 5,
            'PROCEDURES': 6, 'DOCUMENTS': 7, 'POLLUTANT': 8,
            'REQUIREMENTS': 9, 'AUTHORITIES': 10, 'REGION': 11
        }
        
        self.relation_types = {
            'REQUIRED': 0, 'COMPLY_WITH': 1, 'GENERATES': 2,
            'HAVE': 3, 'FOLLOW': 4, 'IMPLEMENT': 5,
            'REGULATES': 6, 'MONITORS': 7, 'RESTRICT': 8,
            'CARRY_OUT': 9, 'APPLY_TO': 10, 'MANAGES': 11,
            'RELATED_TO': 12
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process annotations
        entities = []
        relations = []
        for ann in item['annotations']:
            if ann['type'] == 'entity':
                entities.append({
                    'label': self.entity_types[ann['label']],
                    'start': ann['start'],
                    'end': ann['end']
                })
            elif ann['type'] == 'relation':
                relations.append({
                    'from': ann['from_entity'],
                    'to': ann['to_entity'],
                    'label': self.relation_types[ann['label']]
                })
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'entities': entities,
            'relations': relations
        }


def custom_collate(batch):
    # Pad input_ids and attention_mask
    max_length = max(x['input_ids'].size(0) for x in batch)
    
    padded_input_ids = []
    padded_attention_mask = []
    entities_list = []
    relations_list = []
    
    for item in batch:
        # Pad input_ids
        pad_length = max_length - item['input_ids'].size(0)
        padded_input_ids.append(
            torch.cat([item['input_ids'], 
                      torch.zeros(pad_length, dtype=torch.long)], dim=0)
        )
        
        # Pad attention_mask
        padded_attention_mask.append(
            torch.cat([item['attention_mask'], 
                      torch.zeros(pad_length, dtype=torch.long)], dim=0)
        )
        
        # Store entities and relations as is
        entities_list.append(item['entities'])
        relations_list.append(item['relations'])
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'entities': entities_list,
        'relations': relations_list
    }