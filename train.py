import torch
from torch.utils.data import DataLoader
from models.unirel_model import UniRelModel
from models.data_loader import MaritimeRegulationsDataset, custom_collate
from transformers import AdamW
import torch.nn.functional as F

def calculate_entity_loss(logits, targets):
    """Calculate entity recognition loss"""
    batch_size, seq_len, num_labels = logits.size()
    device = logits.device
    
    # Create target tensor
    target_tensor = torch.zeros((batch_size, seq_len, num_labels), device=device)
    
    # Fill in target tensor based on entity annotations
    for batch_idx, batch_entities in enumerate(targets):
        for entity in batch_entities:
            start, end = entity['start'], entity['end']
            if start < seq_len and end <= seq_len:  # Ensure within sequence length
                target_tensor[batch_idx, start:end, entity['label']] = 1
    
    # Calculate BCE loss for multi-label classification
    loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
    return loss

def calculate_relation_loss(logits, targets):
    """Calculate relation extraction loss"""
    batch_size, seq_len, _, num_labels = logits.size()
    device = logits.device
    
    # Create target tensor
    target_tensor = torch.zeros((batch_size, seq_len, seq_len, num_labels), device=device)
    
    # Fill in target tensor based on relation annotations
    for batch_idx, batch_relations in enumerate(targets):
        for relation in batch_relations:
            from_idx, to_idx = relation['from'], relation['to']
            if from_idx < seq_len and to_idx < seq_len:  # Ensure within sequence length
                target_tensor[batch_idx, from_idx, to_idx, relation['label']] = 1
    
    # Calculate BCE loss for multi-label classification
    loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
    return loss

def train_model(train_file, val_file=None, epochs=10, batch_size=4, learning_rate=2e-5):
    # Initialize dataset
    dataset = MaritimeRegulationsDataset(train_file, max_length=256)  # Reduce sequence length
    
    # Initialize dataloader with smaller batch size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=custom_collate
    )
    
    model = UniRelModel()
    
    # Handle device placement
    if torch.cuda.is_available():
        # Set memory allocation configuration
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop with gradient accumulation
    accumulation_steps = 4  # Accumulate gradients for 4 steps
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            entity_loss = calculate_entity_loss(outputs['entity_logits'], batch['entities'])
            relation_loss = calculate_relation_loss(outputs['relation_logits'], batch['relations'])
            
            loss = (entity_loss + relation_loss) / accumulation_steps
            total_loss += loss.item() * accumulation_steps
            
            # Backward pass with gradient accumulation
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Clear memory
            del outputs, loss, entity_loss, relation_loss
            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

if __name__ == "__main__":
    train_file = r"./preprocessing/data/output/annotated_text.json"
    model = train_model(train_file)
    
    # Save the trained model
    torch.save(model.state_dict(), r"./models/unirel_model.pth")