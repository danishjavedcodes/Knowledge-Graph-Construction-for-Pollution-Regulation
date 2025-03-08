import torch
import torch.nn as nn
import transformers
from torch.nn import functional as F

class UniRelModel(nn.Module):
    def __init__(self, 
                 pretrained_model="bert-base-uncased",
                 num_entity_types=13,  # Your entity types (SYSTEM, EQUIPMENT, etc.)
                 num_relation_types=15):  # Your relation types
        super().__init__()
        
        # Token + Sequence + Position Embedding
        self.bert = transformers.BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        
        # Self-Attention Layers
        self.self_attention = nn.MultiheadAttention(
            embed_dim=768,  # BERT base hidden size
            num_heads=8,
            dropout=0.1
        )
        
        # Entity Recognition Layer
        self.entity_classifier = nn.Linear(768, num_entity_types)
        
        # Relation Extraction Layer
        self.relation_classifier = nn.Linear(768 * 2, num_relation_types)
        
    def forward(self, input_ids, attention_mask):
        # Embedding Layer
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        
        # Self-Attention Mechanism
        attended_output, _ = self.self_attention(
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2)
        )
        attended_output = attended_output.permute(1, 0, 2)
        
        # Entity Recognition
        entity_logits = self.entity_classifier(self.dropout(attended_output))
        
        # Relation Extraction
        batch_size, seq_len, hidden_size = attended_output.size()
        
        # Create entity pairs for relation extraction
        entity_pairs = torch.cat([
            attended_output.unsqueeze(2).expand(-1, -1, seq_len, -1),
            attended_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
        ], dim=-1)
        
        # Relation Classification
        relation_logits = self.relation_classifier(
            self.dropout(entity_pairs)
        )
        
        return {
            'entity_logits': entity_logits,
            'relation_logits': relation_logits
        }