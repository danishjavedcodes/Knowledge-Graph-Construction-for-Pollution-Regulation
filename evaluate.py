import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from train import MaritimeDataset, KGBertModel
from transformers import RobertaTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path, num_entity_types, num_relation_types, device):
    model = KGBertModel(num_entity_types, num_relation_types).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    predictions = {
        'entity1': [],
        'entity2': [],
        'relation': []
    }
    true_labels = {
        'entity1': [],
        'entity2': [],
        'relation': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            entity1_logits, entity2_logits, relation_logits = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['entity1_pos'],
                batch['entity2_pos']
            )
            
            # Get predictions
            predictions['entity1'].extend(entity1_logits.argmax(dim=-1).cpu().numpy())
            predictions['entity2'].extend(entity2_logits.argmax(dim=-1).cpu().numpy())
            predictions['relation'].extend(relation_logits.argmax(dim=-1).cpu().numpy())
            
            # Get true labels
            true_labels['entity1'].extend(batch['entity1_type'].cpu().numpy())
            true_labels['entity2'].extend(batch['entity2_type'].cpu().numpy())
            true_labels['relation'].extend(batch['relation_type'].cpu().numpy())
    
    return predictions, true_labels

def print_metrics(predictions, true_labels, entity_types, relation_types):
    tasks = {
        'Entity 1': ('entity1', entity_types),
        'Entity 2': ('entity2', entity_types),
        'Relations': ('relation', relation_types)
    }
    
    # Initialize metrics storage
    metrics_table = {
        'Entity 1': {},
        'Entity 2': {},
        'Relations': {},
        'Overall': {}
    }
    
    print("\n=== Model Evaluation Results ===\n")
    print(f"{'Task':<12} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1 Score':>8}")
    print("-" * 55)
    
    for task_name, (task_key, label_names) in tasks.items():
        # Convert numeric labels back to text for readable report
        label_map = {i: label for i, label in enumerate(label_names)}
        true_labels_text = [label_map[label] for label in true_labels[task_key]]
        pred_labels_text = [label_map[label] for label in predictions[task_key]]
        
        # Get classification report as dictionary
        report = classification_report(true_labels_text, pred_labels_text, 
                                    zero_division=0, output_dict=True)
        
        # Store metrics
        metrics_table[task_name] = {
            'accuracy': accuracy_score(true_labels[task_key], predictions[task_key]),
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        # Print row
        print(f"{task_name:<12} | {metrics_table[task_name]['accuracy']:8.4f} | "
              f"{metrics_table[task_name]['precision']:9.4f} | "
              f"{metrics_table[task_name]['recall']:6.4f} | "
              f"{metrics_table[task_name]['f1']:8.4f}")
    
    # Calculate overall metrics
    metrics_table['Overall'] = {
        'accuracy': np.mean([m['accuracy'] for m in list(metrics_table.values())[:-1]]),
        'precision': np.mean([m['precision'] for m in list(metrics_table.values())[:-1]]),
        'recall': np.mean([m['recall'] for m in list(metrics_table.values())[:-1]]),
        'f1': np.mean([m['f1'] for m in list(metrics_table.values())[:-1]])
    }
    
    print("-" * 55)
    print(f"{'Overall':<12} | {metrics_table['Overall']['accuracy']:8.4f} | "
          f"{metrics_table['Overall']['precision']:9.4f} | "
          f"{metrics_table['Overall']['recall']:6.4f} | "
          f"{metrics_table['Overall']['f1']:8.4f}")
    
    return metrics_table

def plot_evaluation_results(metrics_table, save_dir='d:/polution/KG/metrics'):
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    tasks = ['Entity 1', 'Entity 2', 'Relations']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create bar plot for all metrics
    x = np.arange(len(tasks))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [metrics_table[task][metric] for task in tasks]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Tasks')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics by Task')
    plt.xticks(x + width*1.5, tasks)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrices for each task
    results_df = pd.read_csv(f'{save_dir}/evaluation_results.csv')
    tasks_data = [
        ('Entity1', results_df['Entity1_True'], results_df['Entity1_Pred']),
        ('Entity2', results_df['Entity2_True'], results_df['Entity2_Pred']),
        ('Relation', results_df['Relation_True'], results_df['Relation_Pred'])
    ]
    
    for task_name, true_labels, pred_labels in tasks_data:
        plt.figure(figsize=(10, 8))
        unique_labels = sorted(list(set(true_labels) | set(pred_labels)))
        conf_matrix = pd.crosstab(true_labels, pred_labels)
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd')
        plt.title(f'{task_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{save_dir}/{task_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # First load training data to get the original entity and relation types
    train_data = pd.read_csv('d:/polution/KG/data/data.csv')
    entity_types = sorted(list(set(train_data['entity1_type'].unique()) | set(train_data['entity2_type'].unique())))
    relation_types = sorted(train_data['relation_type'].unique())
    
    # Load the test data
    test_data = pd.read_csv('d:/polution/KG/data/test.csv')
    
    # Create type to index mappings using training data types
    entity_type_dict = {t: i for i, t in enumerate(entity_types)}
    relation_type_dict = {t: i for i, t in enumerate(relation_types)}
    
    # Initialize tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_dataset = MaritimeDataset(test_data, tokenizer, device, entity_type_dict, relation_type_dict)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Load the trained model
    model = load_model(
        'd:/polution/KG/models/kg_bert_model.pt',
        len(entity_types),
        len(relation_types),
        device
    )
    
    # Evaluate the model
    predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Print detailed metrics
    metrics_table = print_metrics(predictions, true_labels, entity_types, relation_types)
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'Entity1_True': [entity_types[label] for label in true_labels['entity1']],
        'Entity1_Pred': [entity_types[label] for label in predictions['entity1']],
        'Entity2_True': [entity_types[label] for label in true_labels['entity2']],
        'Entity2_Pred': [entity_types[label] for label in predictions['entity2']],
        'Relation_True': [relation_types[label] for label in true_labels['relation']],
        'Relation_Pred': [relation_types[label] for label in predictions['relation']]
    })
    results_df.to_csv('d:/polution/KG/metrics/evaluation_results.csv', index=False)
    
    # Plot evaluation results
    plot_evaluation_results(metrics_table)

if __name__ == '__main__':
    main()