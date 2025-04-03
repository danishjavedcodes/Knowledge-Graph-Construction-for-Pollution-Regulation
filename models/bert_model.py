import torch
import torch.nn as nn
from transformers import RobertaModel

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
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, num_entity_types)
        )
        
        # Relation classification layers
        self.relation_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, num_relation_types)
        )

    def forward(self, input_ids, attention_mask, entity1_pos, entity2_pos):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        batch_size = sequence_output.size(0)
        entity1_repr = torch.stack([sequence_output[i, entity1_pos[i]] for i in range(batch_size)])
        entity2_repr = torch.stack([sequence_output[i, entity2_pos[i]] for i in range(batch_size)])
        
        entity1_type_logits = self.entity_classifier(entity1_repr)
        entity2_type_logits = self.entity_classifier(entity2_repr)
        
        relation_input = torch.cat([entity1_repr, entity2_repr], dim=-1)
        relation_logits = self.relation_classifier(relation_input)
        
        return entity1_type_logits, entity2_type_logits, relation_logits