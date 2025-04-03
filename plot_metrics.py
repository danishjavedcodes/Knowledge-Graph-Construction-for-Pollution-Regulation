import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
train_metrics = pd.read_csv('d:/polution/KG/metrics/train_metrics.csv')
val_metrics = pd.read_csv('d:/polution/KG/metrics/val_metrics.csv')

# Set style
plt.style.use('default')  
sns.set_theme() 
sns.set_palette("husl")

def plot_comparison(train_data, val_data, metrics, title, filename):
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_data) + 1)
    
    for metric in metrics:
        plt.plot(epochs, train_data[metric], '-o', label=f'Train {metric}', markersize=4)
        plt.plot(epochs, val_data[metric], '--o', label=f'Val {metric}', markersize=4)
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'd:/polution/KG/metrics/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 1. Loss comparison
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_metrics) + 1), train_metrics['loss'], '-o', label='Train Loss', markersize=4)
plt.plot(range(1, len(val_metrics) + 1), val_metrics['loss'], '--o', label='Validation Loss', markersize=4)
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('d:/polution/KG/metrics/loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Entity 1 metrics comparison
plot_comparison(
    train_metrics, 
    val_metrics,
    ['entity1_precision', 'entity1_recall', 'entity1_f1'],
    'Entity 1 - Training vs Validation Metrics',
    'entity1_metrics'
)

# 3. Entity 2 metrics comparison
plot_comparison(
    train_metrics, 
    val_metrics,
    ['entity2_precision', 'entity2_recall', 'entity2_f1'],
    'Entity 2 - Training vs Validation Metrics',
    'entity2_metrics'
)

# 4. Relation metrics comparison
plot_comparison(
    train_metrics, 
    val_metrics,
    ['relation_precision', 'relation_recall', 'relation_f1'],
    'Relation - Training vs Validation Metrics',
    'relation_metrics'
)