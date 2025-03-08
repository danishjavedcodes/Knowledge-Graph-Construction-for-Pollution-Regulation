import json
import re
import spacy

class AutoLabeler:
    def __init__(self):
        # Load SpaCy model for English
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define entity patterns
        self.entity_patterns = {
            'SYSTEM': r'(system|equipment|machinery|apparatus)',
            'EQUIPMENT': r'(tanker|ship|vessel|filtering equipment|separator)',
            'SPECIFICATION': r'(GT|tons|metres|capacity|volume)',
            'DISPOSAL': r'(discharge|disposal|cleaning|washing)',
            'ACTIVITIES': r'(survey|inspection|operation|maintenance)',
            'PENALTIES': r'(penalty|fine|offense|violation)',
            'PROCEDURES': r'(procedure|process|method|operation)',
            'DOCUMENTS': r'(Certificate|Record Book|documentation)',
            'POLLUTANT': r'(oil|sludge|oily mixture|effluent)',
            'REQUIREMENTS': r'(requirement|regulation|provision)',
            'AUTHORITIES': r'(Authority|Registrar|surveyor|Government)',
            'REGION': r'(Zanzibar|Tanzania|territorial sea)'
        }
        
        # Define relation patterns
        self.relation_patterns = {
            'REQUIRED': r'(require|must|shall|need)',
            'COMPLY_WITH': r'(comply|accordance|conformity)',
            'GENERATES': r'(generate|produce|create)',
            'HAVE': r'(have|has|contain)',
            'FOLLOW': r'(follow|according)',
            'IMPLEMENT': r'(implement|execute|carry out)',
            'REGULATES': r'(regulate|control|govern)',
            'MONITORS': r'(monitor|observe|track)',
            'RESTRICT': r'(restrict|limit|constrain)',
            'CARRY_OUT': r'(perform|conduct|undertake)',
            'APPLY_TO': r'(apply|relevant|applicable)',
            'MANAGES': r'(manage|handle|operate)',
            'RELATED_TO': r'(related|connected|associated)'
        }
    def process_file(self, input_path, output_path):
        # Read JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        # Process each text segment
        for item in data:
            text = item['text']
            annotations = self.annotate_text(text)
            item['annotations'] = annotations
    
        # Save annotated data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def annotate_text(self, text):
        annotations = []
        doc = self.nlp(text)
        entity_map = {}  # To store entity positions
    
        # First pass: Entity recognition
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    'type': 'entity',
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group()
                }
                annotations.append(entity)
                entity_map[match.start()] = len(annotations) - 1
    
        # Second pass: Relation extraction
        # Fix: Sort entities by their start position using a key function
        entities = [(e['start'], e) for e in annotations if e['type'] == 'entity']
        entities.sort(key=lambda x: x[0])  # Sort based on start position
        
        for i, (start1, ent1) in enumerate(entities):
            for j, (start2, ent2) in enumerate(entities[i+1:], i+1):
                # Check if entities are within reasonable distance
                if abs(start2 - start1) > 50:
                    continue
                    
                # Get text between entities
                text_between = text[ent1['end']:ent2['start']].lower()
                
                # Check for relation patterns
                relation_type = None
                for rel_type, pattern in self.relation_patterns.items():
                    if re.search(pattern, text_between, re.IGNORECASE):
                        relation_type = rel_type
                        break
                
                if relation_type:
                    relation = {
                        'type': 'relation',
                        'from_entity': entity_map[start1],
                        'to_entity': entity_map[start2],
                        'label': relation_type,
                        'text': text_between.strip()
                    }
                    annotations.append(relation)

        return annotations

    def check_relation(self, doc, ent1, ent2):
        # Simple proximity-based relation check
        return True if abs(doc.text.index(ent1) - doc.text.index(ent2)) < 50 else False

    def determine_relation_type(self, doc, ent1, ent2):
        # Basic relation type determination
        text_between = doc.text[doc.text.index(ent1):doc.text.index(ent2)]
        
        if 'require' in text_between.lower():
            return 'REQUIRED'
        elif 'comply' in text_between.lower():
            return 'COMPLY_WITH'
        elif 'regulate' in text_between.lower():
            return 'REGULATES'
        else:
            return 'RELATED_TO'

# Usage
if __name__ == "__main__":
    labeler = AutoLabeler()
    input_path = r"./cleaned_text.json"
    output_path = r"./annotated_text.json"
    labeler.process_file(input_path, output_path)