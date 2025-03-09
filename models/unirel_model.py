import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class UniRelModel(nn.Module):
    def __init__(self, num_entity_types=12, num_relation_types=13):
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = 768
        
        # Reduce attention heads and add gradient checkpointing
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,  # Reduced from 8
            dropout=0.1,
            batch_first=True
        )
        
        self.entity_classifier = nn.Linear(self.hidden_size, num_entity_types)
        self.relation_classifier = nn.Linear(self.hidden_size * 2, num_relation_types)
        
        # Enable gradient checkpointing
        self.bert.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        batch_size, seq_len, hidden_size = sequence_output.size()
        
        # Entity Recognition
        entity_logits = self.entity_classifier(self.dropout(sequence_output))
        
        # Relation Extraction with smaller chunks and memory optimization
        chunk_size = 16  # Further reduced chunk size
        relation_logits = torch.zeros(
            (batch_size, seq_len, seq_len, self.relation_classifier.out_features),
            device=sequence_output.device
        )
        
        # Process chunks with memory optimization
        for i in range(0, seq_len, chunk_size):
            i_end = min(i + chunk_size, seq_len)
            chunk_i = sequence_output[:, i:i_end, :]
            
            for j in range(0, seq_len, chunk_size):
                j_end = min(j + chunk_size, seq_len)
                chunk_j = sequence_output[:, j:j_end, :]
                
                # Process smaller chunks
                with torch.cuda.amp.autocast():
                    chunk_pairs = torch.cat([
                        chunk_i.unsqueeze(2).expand(-1, -1, j_end-j, -1),
                        chunk_j.unsqueeze(1).expand(-1, i_end-i, -1, -1)
                    ], dim=-1)
                    
                    chunk_relations = self.relation_classifier(self.dropout(chunk_pairs))
                    relation_logits[:, i:i_end, j:j_end] = chunk_relations
                
                # Clear memory immediately
                del chunk_pairs, chunk_relations
                torch.cuda.empty_cache()
            
            # Clear chunk_i memory
            del chunk_i
            torch.cuda.empty_cache()
        
        return {
            'entity_logits': entity_logits,
            'relation_logits': relation_logits
        }