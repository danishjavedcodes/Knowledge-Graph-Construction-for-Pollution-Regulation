import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
from unirel_model import UniRelModel

class PollutionDataset(Dataset):
    def __init__(self, json_file, tokenizer_name="bert-base-uncased"):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
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
            max_length=512,
            return_tensors='pt'
        )
        
        # Process annotations
        entities = []
        relations = []
        for ann in item['annotations']:
            if ann['type'] == 'entity':
                entities.append({
                    'label': ann['label'],
                    'start': ann['start'],
                    'end': ann['end']
                })
            elif ann['type'] == 'relation':
                relations.append({
                    'from': ann['from_entity'],
                    'to': ann['to_entity'],
                    'label': ann['label']
                })
                
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'entities': entities,
            'relations': relations
        }

def train_model():
    # Initialize model and dataset
    model = UniRelModel()
    dataset = PollutionDataset(r"c:\Users\Danish Javed\Desktop\polution\annotated_text.json")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            
            # Calculate loss
            entity_loss = calculate_entity_loss(outputs['entity_logits'], batch['entities'])
            relation_loss = calculate_relation_loss(outputs['relation_logits'], batch['relations'])
            total_loss = entity_loss + relation_loss
            
            total_loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train_model()