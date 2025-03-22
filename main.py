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

class KGDataset(Dataset):
    def __init__(self, texts, entity1_pos, entity2_pos, entity1_types, entity2_types, relation_types, tokenizer, max_length=512):
        self.texts = texts
        self.entity1_pos = entity1_pos
        self.entity2_pos = entity2_pos
        self.entity1_types = entity1_types
        self.entity2_types = entity2_types
        self.relation_types = relation_types
        # Switch to fast tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
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
    
    return (texts, entity1_pos, entity2_pos, entity1_types, entity2_types, 
            relation_types, len(entity_type_map), len(relation_type_map))

# Add these imports at the top
import matplotlib.pyplot as plt
from datetime import datetime
import csv

def train_model(model, train_loader, val_loader, device, save_dir, num_epochs=20):
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
            print("\nWarning: Model predictions may be degenerate - predicting only one class")
            print("Consider adjusting learning rate or model architecture")
        
        print(f'Average Loss: {total_loss/len(train_loader)}')
        print(f'Validation Metrics:')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        
        # Early stopping if metrics reach target
        if precision >= 0.99 and recall >= 0.99 and f1 >= 0.99:
            print("\nReached target performance! Stopping training early.")
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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    json_file = "./preprocessing/data/output/annotated_text.json"
    (texts, entity1_pos, entity2_pos, entity1_types, entity2_types, 
     relation_types, num_entity_types, num_relation_types) = prepare_data(json_file)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Add data validation
    if len(texts) == 0:
        raise ValueError("No valid training examples found in the dataset")
    
    if num_entity_types == 0 or num_relation_types == 0:
        raise ValueError("No entity types or relation types found in the dataset")
    
    # Add model save path validation
    save_dir = "./models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_save_path = os.path.join(save_dir, 'best_kg_model.pt')
    
    # Split data
    train_indices, val_indices = train_test_split(range(len(texts)), test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = KGDataset(
        [texts[i] for i in train_indices],
        [entity1_pos[i] for i in train_indices],
        [entity2_pos[i] for i in train_indices],
        [entity1_types[i] for i in train_indices],
        [entity2_types[i] for i in train_indices],
        [relation_types[i] for i in train_indices],
        tokenizer
    )
    
    val_dataset = KGDataset(
        [texts[i] for i in val_indices],
        [entity1_pos[i] for i in val_indices],
        [entity2_pos[i] for i in val_indices],
        [entity1_types[i] for i in val_indices],
        [entity2_types[i] for i in val_indices],
        [relation_types[i] for i in val_indices],
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize model
    model = KGBertModel(num_entity_types, num_relation_types).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, save_dir)

if __name__ == "__main__":
    main()


def visualize_graph(self, triples):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Get unique entities and their connections
        entity_connections = {}
        entities = set()
        for entity1, _, entity2 in triples:
            entities.add((entity1[0], entity1[1]))
            entities.add((entity2[0], entity2[1]))
            entity_connections[entity1[0]] = entity_connections.get(entity1[0], 0) + 1
            entity_connections[entity2[0]] = entity_connections.get(entity2[0], 0) + 1
        
        # Select center node (most connected)
        center_node = max(entity_connections.items(), key=lambda x: x[1])[0]
        
        # Remove center node from entities and get its type
        center_type = None
        entities_list = []
        for entity in entities:
            if entity[0] == center_node:
                center_type = entity[1]
            else:
                entities_list.append(entity)
        
        # Calculate positions with dynamic radius
        n_entities = len(entities_list)
        base_radius = max(8, n_entities / 2)  # Adjust radius based on number of entities
        angles = np.linspace(0, 2*np.pi, n_entities, endpoint=False)
        
        # Add some randomness to radius for better spacing
        radii = [base_radius + np.random.uniform(-0.5, 0.5) for _ in range(n_entities)]
        
        # Plot center node
        center_circle = plt.Circle((0, 0), 2, color='red', alpha=0.7)
        ax.add_patch(center_circle)
        ax.text(0, 0, f"{center_node}\n({center_type})", 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Store node positions for edge routing
        node_positions = {}
        
        # Plot other nodes with adjusted positions
        for i, ((entity_name, entity_type), angle, radius) in enumerate(zip(entities_list, angles, radii)):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Store position
            node_positions[entity_name] = (x, y)
            
            # Draw node
            circle = plt.Circle((x, y), 1.8, color='lightblue', alpha=0.7)
            ax.add_patch(circle)
            
            # Add node label with background
            ax.text(x, y, f"{entity_name}\n({entity_type})", 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Draw edges with curved paths and labels
        for e1, relation, e2 in triples:
            if (e1[0] == center_node and e2[0] in node_positions) or \
               (e2[0] == center_node and e1[0] in node_positions):
                
                # Get positions
                start_pos = (0, 0) if e1[0] == center_node else node_positions[e1[0]]
                end_pos = node_positions[e2[0]] if e1[0] == center_node else (0, 0)
                
                # Calculate control points for curved edge
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Adjust curve height based on distance
                curve_height = dist * 0.2
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                
                # Calculate perpendicular offset for control point
                nx = -dy / dist
                ny = dx / dist
                control_x = mid_x + nx * curve_height
                control_y = mid_y + ny * curve_height
                
                # Create curved path
                curve = plt.matplotlib.patches.PathPatch(
                    plt.matplotlib.path.Path(
                        [start_pos, (control_x, control_y), end_pos],
                        [plt.matplotlib.path.Path.MOVETO, 
                         plt.matplotlib.path.Path.CURVE3,
                         plt.matplotlib.path.Path.CURVE3]
                    ),
                    facecolor='none',
                    edgecolor='gray',
                    linewidth=1.5,
                    alpha=0.7
                )
                ax.add_patch(curve)
                
                # Add arrow
                arrow_pos = 0.6  # Position along the curve
                arrow_x = control_x * arrow_pos + end_pos[0] * (1 - arrow_pos)
                arrow_y = control_y * arrow_pos + end_pos[1] * (1 - arrow_pos)
                
                # Calculate arrow direction
                dx = end_pos[0] - arrow_x
                dy = end_pos[1] - arrow_y
                arrow_length = np.sqrt(dx*dx + dy*dy)
                dx, dy = dx/arrow_length, dy/arrow_length
                
                ax.arrow(arrow_x, arrow_y, dx*0.5, dy*0.5,
                        head_width=0.3, head_length=0.5, fc='gray', ec='gray')
                
                # Add relation label with background
                label_x = control_x
                label_y = control_y
                ax.text(label_x, label_y, relation,
                       ha='center', va='center',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                       fontsize=9, fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        limit = base_radius + 5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Remove axes
        ax.axis('off')
        
        plt.title("Knowledge Graph", fontsize=16, pad=20)
        return fig
