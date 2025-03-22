import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm, trange
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import RobertaTokenizerFast
import os
# Add these imports at the top
import warnings
import logging
from transformers import logging as transformers_logging

# Suppress warnings
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class KGBertModel(nn.Module):
    def __init__(self, num_entity_types, num_relation_types):
        super(KGBertModel, self).__init__()
        # Using RoBERTa base model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.hidden_size = 768  # RoBERTa base hidden size
        
        # Freeze most of the RoBERTa layers
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # Unfreeze only the top 2 transformer layers for fine-tuning
        for layer in self.roberta.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Entity type classification layers
        self.entity_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_entity_types)
        )
        
        # Relation classification layers
        # Concatenating representations of both entities
        self.relation_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_relation_types)
        )

    def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
        # Get RoBERTa outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract entity representations using their positions
        batch_size = sequence_output.size(0)
        entity1_repr = torch.stack([sequence_output[i, entity1_pos[i]] for i in range(batch_size)])
        entity2_repr = torch.stack([sequence_output[i, entity2_pos[i]] for i in range(batch_size)])
        
        # Entity type classification
        entity1_type_logits = self.entity_classifier(entity1_repr)
        entity2_type_logits = self.entity_classifier(entity2_repr)
        
        # Relation classification
        # Concatenate entity representations
        relation_input = torch.cat([entity1_repr, entity2_repr], dim=-1)
        relation_logits = self.relation_classifier(relation_input)
        
        return entity1_type_logits, entity2_type_logits, relation_logits

# Add the KGDataset class definition here or import it correctly
class KGDataset(Dataset):
    def __init__(self, texts, entity1_pos, entity2_pos, entity1_types, entity2_types, relation_types, tokenizer, max_length=512):
        self.texts = texts
        self.entity1_pos = entity1_pos
        self.entity2_pos = entity2_pos
        self.entity1_types = entity1_types
        self.entity2_types = entity2_types
        self.relation_types = relation_types
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text with offset mapping
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Get offset mapping
        offset_mapping = encoding.offset_mapping[0].numpy()
        
        # Map character positions to token positions
        def char_to_token_position(char_pos):
            for idx, (start, end) in enumerate(offset_mapping):
                if start <= char_pos < end:
                    return idx
            return 1  # Default to first non-special token if not found
        
        entity1_token_pos = char_to_token_position(self.entity1_pos[idx])
        entity2_token_pos = char_to_token_position(self.entity2_pos[idx])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'entity1_pos': torch.tensor(entity1_token_pos),
            'entity2_pos': torch.tensor(entity2_token_pos),
            'entity1_type': torch.tensor(self.entity1_types[idx]),
            'entity2_type': torch.tensor(self.entity2_types[idx]),
            'relation_type': torch.tensor(self.relation_types[idx])
        }

def prepare_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    texts = []
    entity1_pos = []
    entity2_pos = []
    entity1_types = []
    entity2_types = []
    relation_types = []
    
    entity_type_map = {}
    relation_type_map = {}
    
    for item in data:
        # Process relations
        for relation in item.get('annotations', []):
            if relation.get('type') == 'relation':
                # Get entities involved in relation
                from_entity = relation.get('from_entity')
                to_entity = relation.get('to_entity')
                relation_label = relation.get('label')
                
                if from_entity is not None and to_entity is not None:
                    # Add text
                    texts.append(item['text'])
                    
                    # Add entity positions
                    entity1_pos.append(from_entity)
                    entity2_pos.append(to_entity)
                    
                    # Map entity types
                    for entity in item['annotations']:
                        if entity.get('type') == 'entity':
                            entity_type = entity.get('label')
                            if entity_type not in entity_type_map:
                                entity_type_map[entity_type] = len(entity_type_map)
                    
                    # Map relation types
                    if relation_label not in relation_type_map:
                        relation_type_map[relation_label] = len(relation_type_map)
                    
                    # Add types
                    entity1_types.append(entity_type_map[entity_type])
                    entity2_types.append(entity_type_map[entity_type])
                    relation_types.append(relation_type_map[relation_label])
    
    # Save mapping files
    model_dir = os.path.dirname(r"d:/polution/Knowledge-Graph-Construction-for-Pollution-Regulation/models/best_kg_model.pt")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'entity_type_map.json'), 'w') as f:
        json.dump(entity_type_map, f, indent=2)
    with open(os.path.join(model_dir, 'relation_type_map.json'), 'w') as f:
        json.dump(relation_type_map, f, indent=2)
    
    return (texts, entity1_pos, entity2_pos, entity1_types, entity2_types, 
            relation_types, len(entity_type_map), len(relation_type_map))

# Add these imports at the top
import matplotlib.pyplot as plt
from datetime import datetime
import csv

def train_model(model, train_loader, val_loader, device, save_dir, num_epochs=50):
    # Initialize optimizer and criterion
    bert_params = [p for n, p in model.named_parameters() if 'roberta' in n and p.requires_grad]
    custom_params = [p for n, p in model.named_parameters() if 'roberta' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': 1e-5},
        {'params': custom_params, 'lr': 2e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize metrics storage
    best_f1 = 0
    train_losses = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(save_dir, f'training_logs_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Define best model path
    best_model_path = os.path.join(save_dir, 'best_kg_model.pt')
    
    # Create CSV log file
    csv_path = os.path.join(log_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Precision', 'Val Recall', 'Val F1'])
    
    # Training loop
    epoch_iterator = trange(num_epochs, desc="Training")
    for epoch in epoch_iterator:
        model.train()
        total_loss = 0
        
        # Add progress bar for batches
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in batch_iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity1_pos = batch['entity1_pos'].to(device)
            entity2_pos = batch['entity2_pos'].to(device)
            entity1_type = batch['entity1_type'].to(device)
            entity2_type = batch['entity2_type'].to(device)
            relation_type = batch['relation_type'].to(device)
            
            # Forward pass
            entity1_logits, entity2_logits, relation_logits = model(
                input_ids, attention_mask, entity1_pos, entity2_pos
            )
            
            # Calculate loss
            loss = (criterion(entity1_logits, entity1_type) + 
                   criterion(entity2_logits, entity2_type) + 
                   criterion(relation_logits, relation_type))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Update batch progress bar with current loss
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        # Add progress bar for validation
        val_iterator = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for batch in val_iterator:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity1_pos = batch['entity1_pos'].to(device)
                entity2_pos = batch['entity2_pos'].to(device)
                
                _, _, relation_logits = model(input_ids, attention_mask, entity1_pos, entity2_pos)
                
                val_preds.extend(torch.argmax(relation_logits, dim=1).cpu().numpy())
                val_labels.extend(batch['relation_type'].numpy())
        
        # Calculate metrics with zero_division parameter
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, 
            average='weighted',
            zero_division=0  # Explicitly handle zero division case
        )
        
        # Add check for valid predictions
        if precision == 0 and recall == 0 and f1 == 0:
            print("/nWarning: Model predictions may be degenerate - predicting only one class")
            print("Consider adjusting learning rate or model architecture")
        
        print(f'Average Loss: {total_loss/len(train_loader)}')
        print(f'Validation Metrics:')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        
        # Early stopping if metrics reach target
        if precision >= 0.99 and recall >= 0.99 and f1 >= 0.99:
            print("/nReached target performance! Stopping training early.")
            # Save final model state
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'train_losses': train_losses,
                    'val_precisions': val_precisions,
                    'val_recalls': val_recalls,
                    'val_f1s': val_f1s,
                }, best_model_path)
            break
            
        # After calculating metrics, store them
        avg_loss = total_loss/len(train_loader)
        train_losses.append(avg_loss)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        
        # Log metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, precision, recall, f1])
        
        # Create and save plots
        plt.figure(figsize=(12, 8))
        epochs_range = range(1, epoch + 2)
        
        # Plot training loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation metrics
        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, val_precisions, 'g-', label='Precision')
        plt.plot(epochs_range, val_recalls, 'r-', label='Recall')
        plt.plot(epochs_range, val_f1s, 'y-', label='F1 Score')
        plt.title('Validation Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
        plt.close()
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'train_losses': train_losses,
                'val_precisions': val_precisions,
                'val_recalls': val_recalls,
                'val_f1s': val_f1s,
            }, os.path.join(log_dir, 'best_kg_model.pt'))
        
        scheduler.step()
    
    # Gradient clipping (should be inside the training loop, before optimizer.step())
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"./best_kg_model.pt"
    data_path = r"./preprocessing/data/output/annotated_text.json"
    
    # First prepare data to generate mapping files
    (texts, entity1_pos, entity2_pos, entity1_types, entity2_types, 
     relation_types, num_entity_types, num_relation_types) = prepare_data(data_path)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    # Create test dataset (using all data for evaluation)
    test_dataset = KGDataset(
        texts, entity1_pos, entity2_pos, 
        entity1_types, entity2_types, relation_types,
        tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load the trained model
    model = KGBertModel(num_entity_types, num_relation_types)
    try:
        # Try loading as state dict
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If not a state dict, load directly
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Get class names from the mapping files
    model_dir = './models'
    with open(os.path.join(model_dir, 'entity_type_map.json')) as f:
        entity_type_map = json.load(f)
    with open(os.path.join(model_dir, 'relation_type_map.json')) as f:
        relation_type_map = json.load(f)
    
    entity_classes = list(entity_type_map.keys())
    relation_classes = list(relation_type_map.keys())
    
    # Run evaluation
    def evaluate(model, test_loader, device, entity_classes, relation_classes):
        model.eval()
        entity1_preds = []
        entity2_preds = []
        relation_preds = []
        entity1_labels = []
        entity2_labels = []
        relation_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity1_pos = batch['entity1_pos'].to(device)
                entity2_pos = batch['entity2_pos'].to(device)
                
                # Get model predictions
                entity1_logits, entity2_logits, relation_logits = model(
                    input_ids, attention_mask, entity1_pos, entity2_pos
                )
                
                # Convert logits to predictions
                entity1_preds.extend(torch.argmax(entity1_logits, dim=1).cpu().numpy())
                entity2_preds.extend(torch.argmax(entity2_logits, dim=1).cpu().numpy())
                relation_preds.extend(torch.argmax(relation_logits, dim=1).cpu().numpy())
                
                # Store labels
                entity1_labels.extend(batch['entity1_type'].numpy())
                entity2_labels.extend(batch['entity2_type'].numpy())
                relation_labels.extend(batch['relation_type'].numpy())
        
        # Calculate metrics for each task
        results = {}
        
        # Entity 1 metrics
        entity1_metrics = precision_recall_fscore_support(
            entity1_labels, entity1_preds, average='weighted', zero_division=0
        )
        results['Entity1'] = {
            'precision': entity1_metrics[0],
            'recall': entity1_metrics[1],
            'f1': entity1_metrics[2],
            'accuracy': np.mean(np.array(entity1_preds) == np.array(entity1_labels))
        }
        
        # Entity 2 metrics
        entity2_metrics = precision_recall_fscore_support(
            entity2_labels, entity2_preds, average='weighted', zero_division=0
        )
        results['Entity2'] = {
            'precision': entity2_metrics[0],
            'recall': entity2_metrics[1],
            'f1': entity2_metrics[2],
            'accuracy': np.mean(np.array(entity2_preds) == np.array(entity2_labels))
        }
        
        # Relation metrics
        relation_metrics = precision_recall_fscore_support(
            relation_labels, relation_preds, average='weighted', zero_division=0
        )
        results['Relation'] = {
            'precision': relation_metrics[0],
            'recall': relation_metrics[1],
            'f1': relation_metrics[2],
            'accuracy': np.mean(np.array(relation_preds) == np.array(relation_labels))
        }
        
        return results
    
    # Run evaluation
    results = evaluate(model, test_loader, device, entity_classes, relation_classes)
    
    # Print final metrics
    print("/nFinal Evaluation Metrics:")
    for task, metrics in results.items():
        print(f"{task:<10} | Accuracy: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
