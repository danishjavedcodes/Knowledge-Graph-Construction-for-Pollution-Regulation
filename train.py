import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm

class MaritimeDataset(Dataset):
    def __init__(self, data, tokenizer, device, entity_types, relation_types, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.entity_types = entity_types
        self.relation_types = relation_types
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['sentence'])  # Ensure text is string
        
        encoding = self.tokenizer(text, 
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_tensors='pt')
        
        # Safe position calculation
        entity1_start = min(int(row['entity1_start']), len(text))
        entity2_start = min(int(row['entity2_start']), len(text))
        
        entity1_pos = len(self.tokenizer.encode(text[:entity1_start])) - 1
        entity2_pos = len(self.tokenizer.encode(text[:entity2_start])) - 1
        
        # Convert to tensors and move to device
        return {
            'input_ids': encoding['input_ids'].squeeze(0).to(self.device),
            'attention_mask': encoding['attention_mask'].squeeze(0).to(self.device),
            'entity1_pos': torch.tensor(entity1_pos, device=self.device),
            'entity2_pos': torch.tensor(entity2_pos, device=self.device),
            'entity1_type': torch.tensor(self.entity_types[row['entity1_type']], device=self.device),
            'entity2_type': torch.tensor(self.entity_types[row['entity2_type']], device=self.device),
            'relation_type': torch.tensor(self.relation_types[row['relation_type']], device=self.device)
        }

class KGBertModel(torch.nn.Module):
    def __init__(self, num_entity_types, num_relation_types):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.roberta.config.hidden_size
        
        self.entity_classifier = torch.nn.Linear(hidden_size, num_entity_types)
        self.relation_classifier = torch.nn.Linear(hidden_size * 2, num_relation_types)
        
    def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Extract entity representations
        entity1_repr = hidden_states[torch.arange(hidden_states.size(0)), entity1_pos]
        entity2_repr = hidden_states[torch.arange(hidden_states.size(0)), entity2_pos]
        
        # Classify entities
        entity1_logits = self.entity_classifier(entity1_repr)
        entity2_logits = self.entity_classifier(entity2_repr)
        
        # Classify relation
        relation_input = torch.cat([entity1_repr, entity2_repr], dim=-1)
        relation_logits = self.relation_classifier(relation_input)
        
        return entity1_logits, entity2_logits, relation_logits

def calculate_metrics(preds, labels):
    metrics = {}
    for task in ['entity1', 'entity2', 'relation']:
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels[task], 
            preds[task],
            average='weighted',
            zero_division=0
        )
        accuracy = accuracy_score(labels[task], preds[task])
        
        metrics.update({
            f'{task}_precision': precision,
            f'{task}_recall': recall,
            f'{task}_f1': f1,
            f'{task}_accuracy': accuracy
        })
    
    metrics['accuracy'] = np.mean([metrics[f'{task}_accuracy'] for task in ['entity1', 'entity2', 'relation']])
    return metrics

def train_model(model, train_loader, val_loader, device, num_epochs=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    train_logs = []
    val_logs = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = {'entity1': [], 'entity2': [], 'relation': []}
        all_labels = {'entity1': [], 'entity2': [], 'relation': []}
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            
            entity1_logits, entity2_logits, relation_logits = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['entity1_pos'],
                batch['entity2_pos']
            )
            
            loss = (criterion(entity1_logits, batch['entity1_type']) +
                   criterion(entity2_logits, batch['entity2_type']) +
                   criterion(relation_logits, batch['relation_type'])) / 3
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions and labels
            all_preds['entity1'].extend(entity1_logits.argmax(dim=-1).cpu().numpy())
            all_preds['entity2'].extend(entity2_logits.argmax(dim=-1).cpu().numpy())
            all_preds['relation'].extend(relation_logits.argmax(dim=-1).cpu().numpy())
            
            all_labels['entity1'].extend(batch['entity1_type'].cpu().numpy())
            all_labels['entity2'].extend(batch['entity2_type'].cpu().numpy())
            all_labels['relation'].extend(batch['relation_type'].cpu().numpy())
        
        train_metrics = calculate_metrics(all_preds, all_labels)
        train_metrics['loss'] = train_loss / len(train_loader)
        train_logs.append(train_metrics)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = {'entity1': [], 'entity2': [], 'relation': []}
        all_labels = {'entity1': [], 'entity2': [], 'relation': []}
        
        with torch.no_grad():
            for batch in val_loader:
                entity1_logits, entity2_logits, relation_logits = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['entity1_pos'],
                    batch['entity2_pos']
                )
                
                loss = (criterion(entity1_logits, batch['entity1_type']) +
                       criterion(entity2_logits, batch['entity2_type']) +
                       criterion(relation_logits, batch['relation_type'])) / 3
                
                val_loss += loss.item()
                
                # Store predictions and labels
                all_preds['entity1'].extend(entity1_logits.argmax(dim=-1).cpu().numpy())
                all_preds['entity2'].extend(entity2_logits.argmax(dim=-1).cpu().numpy())
                all_preds['relation'].extend(relation_logits.argmax(dim=-1).cpu().numpy())
                
                all_labels['entity1'].extend(batch['entity1_type'].cpu().numpy())
                all_labels['entity2'].extend(batch['entity2_type'].cpu().numpy())
                all_labels['relation'].extend(batch['relation_type'].cpu().numpy())
        
        val_metrics = calculate_metrics(all_preds, all_labels)
        val_metrics['loss'] = val_loss / len(val_loader)
        val_logs.append(val_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        scheduler.step(val_metrics['accuracy'])
    
    return pd.DataFrame(train_logs), pd.DataFrame(val_logs)

def main():
    # Initialize device and check GPU availability
    if not torch.cuda.is_available():
        print("Error: No GPU available. Training requires a GPU.")
        return
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and preprocess data
    data = pd.read_csv('./data/data.csv')
    
    # Get unique entity and relation types
    entity_types = sorted(list(set(data['entity1_type'].unique()) | set(data['entity2_type'].unique())))
    relation_types = sorted(data['relation_type'].unique())
    
    # Create type to index mappings
    entity_type_dict = {t: i for i, t in enumerate(entity_types)}
    relation_type_dict = {t: i for i, t in enumerate(relation_types)}
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize tokenizer and datasets
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_dataset = MaritimeDataset(train_data, tokenizer, device, entity_type_dict, relation_type_dict)
    val_dataset = MaritimeDataset(val_data, tokenizer, device, entity_type_dict, relation_type_dict)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0)
    
    # Initialize model
    model = KGBertModel(len(entity_types), len(relation_types)).to(device)
    
    # Train model
    train_logs, val_logs = train_model(model, train_loader, val_loader, device)
    
    # Save results
    train_logs.to_csv('./metrics/train_metrics.csv', index=False)
    val_logs.to_csv('./metrics/val_metrics.csv', index=False)
    torch.save(model.state_dict(), './models/kg_bert_model.pt')

if __name__ == '__main__':
    main()