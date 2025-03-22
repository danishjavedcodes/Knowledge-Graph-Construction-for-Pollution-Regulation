import json
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import csv

class AutoLabeler:
    def __init__(self):
        # Download required NLTK data
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load SpaCy model for English
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define entity types with numeric IDs
        self.entity_types = {
            'SYSTEM': 0, 'EQUIPMENT': 1, 'SPECIFICATION': 2,
            'DISPOSAL': 3, 'ACTIVITIES': 4, 'PENALTIES': 5,
            'PROCEDURES': 6, 'DOCUMENTS': 7, 'POLLUTANT': 8,
            'REQUIREMENTS': 9, 'AUTHORITIES': 10, 'REGION': 11,
            'SUBJECT REGULATED': 12
        }
        
        # Define relation types with numeric IDs
        self.relation_types = {
            'REQUIRED': 0, 'COMPLY_WITH': 1, 'GENERATES': 2,
            'HAVE': 3, 'FOLLOW': 4, 'IMPLEMENT': 5,
            'REGULATES': 6, 'MONITORS': 7, 'RESTRICT': 8,
            'CARRY_OUT': 9, 'APPLY_TO': 10, 'MANAGES': 11,
            'RELATED_TO': 12, 'INVOLVE_IN': 13, 'EXTABLISHED': 14
        }
        
        # Define entity patterns based on combined.txt
        self.entity_patterns = {
            'SYSTEM': r'(system|equipment|machinery|apparatus|tank|cargo hold)',
            'EQUIPMENT': r'(tanker|ship|vessel|filtering equipment|separator|freight container|portable tank)',
            'SPECIFICATION': r'(GT|tons|metres|capacity|volume|mass|quantity|size)',
            'DISPOSAL': r'(discharge|disposal|cleaning|washing|unload|transfer)',
            'ACTIVITIES': r'(survey|inspection|operation|maintenance|loading|unloading|handling)',
            'PENALTIES': r'(penalty|fine|offense|violation|criminal action)',
            'PROCEDURES': r'(procedure|process|method|operation|practice|instruction)',
            'DOCUMENTS': r'(Certificate|Record Book|documentation|manifest|plan|report)',
            'POLLUTANT': r'(oil|sludge|oily mixture|effluent|waste|dangerous goods|toxic substance)',
            'REQUIREMENTS': r'(requirement|regulation|provision|rule|compliance)',
            'AUTHORITIES': r'(Authority|Registrar|surveyor|Government|port authority|competent authority)',
            'REGION': r'(Zanzibar|Tanzania|territorial sea|port|harbour)',
            'SUBJECT REGULATED': r'(ship owner|operator|company|person|master|employer)'
        }
        
        # Define relation patterns based on combined.txt
        self.relation_patterns = {
            'REQUIRED': r'(require|must|shall|need|obliged|mandatory)',
            'COMPLY_WITH': r'(comply|accordance|conformity|pursuant|according to)',
            'GENERATES': r'(generate|produce|create|cause|result in)',
            'HAVE': r'(have|has|contain|possess|include)',
            'FOLLOW': r'(follow|according|pursuant|based on)',
            'IMPLEMENT': r'(implement|execute|carry out|perform|conduct)',
            'REGULATES': r'(regulate|control|govern|supervise|oversee)',
            'MONITORS': r'(monitor|observe|track|inspect|survey)',
            'RESTRICT': r'(restrict|limit|constrain|prohibit|prevent)',
            'CARRY_OUT': r'(perform|conduct|undertake|execute|carry out)',
            'APPLY_TO': r'(apply|relevant|applicable|pertain|relate to)',
            'MANAGES': r'(manage|handle|operate|administer|control)',
            'RELATED_TO': r'(related|connected|associated|linked|concerning)',
            'INVOLVE_IN': r'(involve|participate|engage|take part)',
            'EXTABLISHED': r'(establish|set up|create|found|institute)'
        }
    def clean_text(self, text):
        """
        Clean the input text by removing special characters, stop words,
        and performing stemming and lemmatization
        """
        # Only remove special characters that aren't part of words
        text = re.sub(r'[^a-zA-Z0-9\s\-_]', ' ', text)
        text = text.lower()
        return text

    def annotate_text(self, text):
        """
        Modified to work with cleaned text
        """
        annotations = []
        doc = self.nlp(text)
        entity_map = {}
        
        # First pass: Entity recognition
        for entity_type, pattern in self.entity_patterns.items():
            # Don't clean the pattern, just make it lowercase
            pattern = pattern.lower()
            matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
            
            for match in matches:
                entity = {
                    'type': 'entity',
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'text': text[match.start():match.end()]  # Use original text
                }
                annotations.append(entity)
                entity_map[match.start()] = len(annotations) - 1
        
        # Second pass: Relation extraction
        entities = [(e['start'], e) for e in annotations if e['type'] == 'entity']
        entities.sort(key=lambda x: x[0])
        
        for i, (start1, ent1) in enumerate(entities):
            for j, (start2, ent2) in enumerate(entities[i+1:], i+1):
                # Increased distance for relation detection
                if abs(start2 - start1) > 100:  # Increased from 50 to 100
                    continue
                    
                # Get text between entities
                text_between = text[ent1['end']:ent2['start']].lower()
                
                # Check for relation patterns
                for rel_type, pattern in self.relation_patterns.items():
                    pattern = pattern.lower()
                    if re.search(pattern, text_between, re.IGNORECASE):
                        relation = {
                            'type': 'relation',
                            'from_entity': entity_map[start1],
                            'to_entity': entity_map[start2],
                            'label': rel_type,
                            'text': text_between.strip()
                        }
                        annotations.append(relation)
                        break

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

    def process_file(self, input_path, output_path):
        # Read JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each text segment
        all_annotations = []
        combined_text = ""
        current_position = 0
        
        for item in data:
            # Clean the text first
            cleaned_text = self.clean_text(item['text'])
            item['cleaned_text'] = cleaned_text
            
            # Get annotations from the cleaned text
            annotations = self.annotate_text(cleaned_text)
            
            # Debug prints
            if annotations:
                print("\nFound annotations:")
                for ann in annotations:
                    if ann['type'] == 'entity':
                        print(f"Entity: {ann['label']} - {ann['text']}")
                    else:
                        entity1 = [a for a in annotations if a['type'] == 'entity'][ann['from_entity']]
                        entity2 = [a for a in annotations if a['type'] == 'entity'][ann['to_entity']]
                        print(f"Relation: {ann['label']} between {entity1['text']} and {entity2['text']}")
            
            # Adjust positions for combined text
            for ann in annotations:
                if ann['type'] == 'entity':
                    ann['start'] += current_position
                    ann['end'] += current_position
            
            all_annotations.extend(annotations)
            combined_text += cleaned_text + " "
            current_position = len(combined_text)
            
            item['annotations'] = annotations
        
        # Save to CSV with specified path
        csv_path = r"d:\polution\Knowledge-Graph-Construction-for-Pollution-Regulation\preprocessing\data\output\annotated_text.csv"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Save CSV file
        self.save_to_csv(all_annotations, combined_text, csv_path)
        
        # Save annotated data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def save_to_csv(self, annotations, text, csv_path):
        """
        Save annotations in CSV format with columns: sentence | entity 1 | entity 2 | relation
        """
        # Get entities and relations
        entities = [a for a in annotations if a['type'] == 'entity']
        relations = [a for a in annotations if a['type'] == 'relation']
        
        # Prepare CSV rows
        rows = []
        for relation in relations:
            try:
                entity1 = entities[relation['from_entity']]
                entity2 = entities[relation['to_entity']]
                
                # Get the sentence containing both entities
                start = min(entity1['start'], entity2['start'])
                end = max(entity1['end'], entity2['end'])
                sentence = text[max(0, start-50):min(len(text), end+50)].strip()
                
                row = {
                    'sentence': sentence,
                    'entity_1': entity1['text'],
                    'entity_2': entity2['text'],
                    'relation': relation['label']
                }
                rows.append(row)
            except (IndexError, KeyError) as e:
                continue
        
        # Write to CSV
        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['sentence', 'entity_1', 'entity_2', 'relation'])
                writer.writeheader()
                writer.writerows(rows)

if __name__ == "__main__":
    labeler = AutoLabeler()
    input_path = r"d:\polution\Knowledge-Graph-Construction-for-Pollution-Regulation\preprocessing\data\output\cleaned_text.json"
    output_path = r"d:\polution\Knowledge-Graph-Construction-for-Pollution-Regulation\preprocessing\data\output\annotated_text.json"
    labeler.process_file(input_path, output_path)