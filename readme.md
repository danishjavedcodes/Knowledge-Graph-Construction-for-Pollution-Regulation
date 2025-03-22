# Knowledge Graph Construction for Pollution Regulation

This project aims to construct knowledge graphs from pollution regulation documents by customizing RoBERTa LLM.

## Project Structure

```
├── preprocessing/
│   ├── data/
│   │   ├── input/
│   │   │   └── process_pdf.py      # PDF preprocessing utilities
│   │   └── output/
│   ├── analyze_labels.py           # Analyze entity and relation labels
│   ├── auto_labeler.py            # Automatic labeling of entities and relations
│   └── pdf_processor.py           # Main PDF processing pipeline
├── models/
│   ├── best_kg_model.pt           # Trained model checkpoint
│   ├── entity_type_map.json       # Entity type mappings
│   └── relation_type_map.json     # Relation type mappings
├── main.py                        # Model building, data splitting and training script
└── evaluate.py                    # Model evaluation script
```

## File Descriptions

### Preprocessing Pipeline
1. **pdf_processor.py**
   - Primary PDF document processor
   - Extracts and cleans text from PDF files
   - Run this first to process raw PDF documents

2. **process_pdf.py**
   - Additional PDF processing utilities
   - Handles text cleaning and sentence tokenization
   - Used by pdf_processor.py

3. **auto_labeler.py**
   - Automatically labels entities and relations in processed text
   - Run after pdf_processor.py
   - Uses NLP techniques to identify potential entities and relationships

4. **analyze_labels.py**
   - Analyzes and validates the labeled data
   - Generates statistics about entity and relation distributions
   - Run after auto_labeler.py to verify labeling quality

### Model Files
1. **entity_type_map.json**
   - Maps entity types to numerical indices
   - Contains 13 entity types (e.g., EQUIPMENT, POLLUTANT, etc.)

2. **relation_type_map.json**
   - Maps relation types to numerical indices
   - Contains 15 relation types (e.g., GENERATES, MANAGES, etc.)

3. **best_kg_model.pt**
   - Trained model checkpoint
   - Contains model weights and training history

### Training and Evaluation
1. **main.py**
   - Main training script
   - Implements the KGBertModel architecture
   - Handles data loading, training, and model saving

2. **evaluate.py**
   - Model evaluation script
   - Computes precision, recall, and F1 scores
   - Generates evaluation plots and metrics

## Usage Order

1. Run preprocessing pipeline:
```bash
python preprocessing/pdf_processor.py
python preprocessing/auto_labeler.py
python preprocessing/analyze_labels.py
```

2. Train the model:
```bash
python main.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

## Entity Types
- EQUIPMENT
- POLLUTANT
- REQUIREMENTS
- REGION
- DOCUMENTS
- AUTHORITIES
- ACTIVITIES
- PROCEDURES
- SYSTEM
- DISPOSAL
- SUBJECT REGULATED
- SPECIFICATION
- PENALTIES

## Relation Types
- GENERATES
- MANAGES
- FOLLOW
- REQUIRED
- ESTABLISHED
- APPLY_TO
- RELATED_TO
- REGULATES
- INVOLVE_IN
- MONITORS
- HAVE
- COMPLY_WITH
- IMPLEMENT
- RESTRICT
- CARRY_OUT
```
