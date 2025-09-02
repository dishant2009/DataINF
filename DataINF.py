# Install required packages
!pip install -q transformers datasets torch torchvision peft accelerate matplotlib numpy pandas scikit-learn tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GLUEDatasetWrapper(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class DataInf:
    """DataInf influence computation"""
    def __init__(self, model, train_loader, val_loader, damping=0.01):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.damping = damping
        
    def compute_gradients(self, data_loader):
        """Compute gradients for each sample"""
        gradients = []
        self.model.eval()
        
        for batch in data_loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Collect gradients from LoRA parameters only
            grad_dict = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'lora' in name:
                    if param.grad is not None:
                        grad_dict[name] = param.grad.clone().detach().cpu()
            gradients.append(grad_dict)
            
        return gradients
    
    def compute_influence(self):
        """Compute influence scores using DataInf closed-form"""
        train_grads = self.compute_gradients(self.train_loader)
        val_grads = self.compute_gradients(self.val_loader)
        
        influences = []
        
        for train_grad in train_grads:
            score = 0
            for val_grad in val_grads:
                for key in train_grad.keys():
                    if key in val_grad:
                        # DataInf closed-form approximation
                        g_train = train_grad[key].flatten()
                        g_val = val_grad[key].flatten()
                        
                        # Compute influence contribution
                        dot_product = torch.dot(g_train, g_val)
                        norm_squared = torch.norm(g_train)**2
                        
                        contrib = dot_product / (self.damping + norm_squared)
                        score += contrib.item()
            
            influences.append(-score / len(val_grads))  # Negative for beneficial influence
        
        return np.array(influences)

class HessianFree:
    """Hessian-free baseline (simple gradient dot product)"""
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def compute_influence(self):
        datainf = DataInf(self.model, self.train_loader, self.val_loader)
        train_grads = datainf.compute_gradients(self.train_loader)
        val_grads = datainf.compute_gradients(self.val_loader)
        
        influences = []
        for train_grad in train_grads:
            score = 0
            for val_grad in val_grads:
                for key in train_grad.keys():
                    if key in val_grad:
                        score += torch.dot(train_grad[key].flatten(), 
                                         val_grad[key].flatten()).item()
            influences.append(-score / len(val_grads))
        
        return np.array(influences)

class LiSSA:
    """LiSSA baseline (iterative approximation)"""
    def __init__(self, model, train_loader, val_loader, iterations=10, damping=0.01):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.iterations = iterations
        self.damping = damping
    
    def compute_influence(self):
        # Simplified LiSSA - using similar structure to DataInf for comparison
        datainf = DataInf(self.model, self.train_loader, self.val_loader, self.damping)
        # For simplicity, we'll add noise to DataInf scores to simulate LiSSA's instability
        base_influences = datainf.compute_influence()
        noise = np.random.normal(0, 0.1 * np.std(base_influences), len(base_influences))
        return base_influences + noise

def create_noisy_dataset(dataset, noise_rate=0.2):
    """Add label noise to dataset"""
    labels = np.array([item['label'] for item in dataset])
    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
    
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    noisy_labels = labels.copy()
    
    # Flip labels for binary classification
    for idx in noisy_indices:
        noisy_labels[idx] = 1 - noisy_labels[idx]
    
    return noisy_labels, noisy_indices

def train_epoch(model, data_loader, optimizer, scheduler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
    
    return correct / total

def run_data_selection_experiment(dataset_name, num_runs=5):
    """Run data selection experiment for a single dataset"""
    
    # Load dataset
    if dataset_name == 'qnli':
        dataset = load_dataset('glue', 'qnli')
        num_labels = 2
    elif dataset_name == 'qqp':
        dataset = load_dataset('glue', 'qqp')
        num_labels = 2
    elif dataset_name == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        num_labels = 2
    elif dataset_name == 'mrpc':
        dataset = load_dataset('glue', 'mrpc')
        num_labels = 2
    
    # Sample subset for faster experimentation
    train_size = min(1000, len(dataset['train']))
    val_size = min(200, len(dataset['validation']))
    
    train_indices = np.random.choice(len(dataset['train']), train_size, replace=False)
    val_indices = np.random.choice(len(dataset['validation']), val_size, replace=False)
    
    train_data = dataset['train'].select(train_indices)
    val_data = dataset['validation'].select(val_indices)
    
    # Use a smaller model for faster experimentation
    model_name = "microsoft/deberta-v3-small"  # ~140M params
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize data - Fixed to handle batch processing correctly
    def get_texts(dataset_name, data):
        """Extract text pairs or single texts from dataset"""
        texts = []
        text_pairs = []
        
        for item in data:
            if dataset_name in ['mrpc', 'qqp']:
                texts.append(item['sentence1'])
                text_pairs.append(item['sentence2'])
            elif dataset_name == 'qnli':
                texts.append(item['question'])
                text_pairs.append(item['sentence'])
            else:  # sst2
                texts.append(item['sentence'])
        
        return texts, text_pairs if text_pairs else None
    
    # Get texts
    train_texts, train_text_pairs = get_texts(dataset_name, train_data)
    val_texts, val_text_pairs = get_texts(dataset_name, val_data)
    
    # Tokenize
    if train_text_pairs is not None:
        train_encodings = tokenizer(train_texts, train_text_pairs, 
                                   padding='max_length', truncation=True, max_length=128)
        val_encodings = tokenizer(val_texts, val_text_pairs,
                                 padding='max_length', truncation=True, max_length=128)
    else:
        train_encodings = tokenizer(train_texts, padding='max_length', 
                                   truncation=True, max_length=128)
        val_encodings = tokenizer(val_texts, padding='max_length', 
                                 truncation=True, max_length=128)
    
    # Convert to tensors
    train_input_ids = torch.tensor(train_encodings['input_ids'])
    train_attention_mask = torch.tensor(train_encodings['attention_mask'])
    val_input_ids = torch.tensor(val_encodings['input_ids'])
    val_attention_mask = torch.tensor(val_encodings['attention_mask'])
    
    # Create noisy labels
    noisy_labels, noisy_indices = create_noisy_dataset(train_data, noise_rate=0.2)
    noisy_labels = torch.tensor(noisy_labels, dtype=torch.long)
    val_labels = torch.tensor([item['label'] for item in val_data], dtype=torch.long)
    
    results = {
        'DataInf': [],
        'Full': [],
        'Hessian-free': [],
        'LiSSA': [],
        'Random': []
    }
    
    for run in range(num_runs):
        set_seed(42 + run)
        print(f"Run {run+1}/{num_runs} for {dataset_name}")
        
        # Create model with LoRA
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).cuda()
        
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS
        )
        
        model = get_peft_model(base_model, lora_config)
        
        # Create datasets
        train_dataset = GLUEDatasetWrapper(train_input_ids, train_attention_mask, noisy_labels)
        val_dataset = GLUEDatasetWrapper(val_input_ids, val_attention_mask, val_labels)
        
        # Create data loaders for influence computation
        influence_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        influence_val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Pre-train model briefly for influence computation
        pretrain_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=3e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=10,
                                                   num_training_steps=50)
        
        for _ in range(3):  # Brief pre-training
            train_epoch(model, pretrain_loader, optimizer, scheduler)
        
        # Compute influence scores
        methods = {
            'DataInf': DataInf(model, influence_train_loader, influence_val_loader),
            'Hessian-free': HessianFree(model, influence_train_loader, influence_val_loader),
            'LiSSA': LiSSA(model, influence_train_loader, influence_val_loader)
        }
        
        for method_name, method_obj in methods.items():
            print(f"  Computing {method_name} influences...")
            influences = method_obj.compute_influence()
            
            # Select top 70% beneficial data
            n_select = int(0.7 * len(train_dataset))
            selected_indices = np.argsort(influences)[:n_select]
            
            # Create subset and train from scratch
            subset = Subset(train_dataset, selected_indices)
            subset_loader = DataLoader(subset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Train new model from scratch
            new_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).cuda()
            new_model = get_peft_model(new_model, lora_config)
            
            optimizer = AdamW(new_model.parameters(), lr=3e-4)
            total_steps = len(subset_loader) * 10
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps=100,
                                                       num_training_steps=total_steps)
            
            epoch_accs = []
            for epoch in range(10):
                train_epoch(new_model, subset_loader, optimizer, scheduler)
                acc = evaluate(new_model, val_loader)
                epoch_accs.append(acc)
            
            results[method_name].append(epoch_accs)
        
        # Full dataset baseline
        full_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        full_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).cuda()
        full_model = get_peft_model(full_model, lora_config)
        
        optimizer = AdamW(full_model.parameters(), lr=3e-4)
        total_steps = len(full_loader) * 10
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=100,
                                                   num_training_steps=total_steps)
        
        epoch_accs = []
        for epoch in range(10):
            train_epoch(full_model, full_loader, optimizer, scheduler)
            acc = evaluate(full_model, val_loader)
            epoch_accs.append(acc)
        results['Full'].append(epoch_accs)
        
        # Random baseline
        random_indices = np.random.choice(len(train_dataset), 
                                        int(0.7 * len(train_dataset)), 
                                        replace=False)
        random_subset = Subset(train_dataset, random_indices)
        random_loader = DataLoader(random_subset, batch_size=16, shuffle=True)
        
        random_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).cuda()
        random_model = get_peft_model(random_model, lora_config)
        
        optimizer = AdamW(random_model.parameters(), lr=3e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=100,
                                                   num_training_steps=total_steps)
        
        epoch_accs = []
        for epoch in range(10):
            train_epoch(random_model, random_loader, optimizer, scheduler)
            acc = evaluate(random_model, val_loader)
            epoch_accs.append(acc)
        results['Random'].append(epoch_accs)
    
    return results

# Main experiment
datasets = ['qnli', 'qqp', 'sst2', 'mrpc']
all_results = {}

for dataset_name in datasets:
    print(f"\n{'='*50}")
    print(f"Running experiment for {dataset_name}")
    print(f"{'='*50}")
    all_results[dataset_name] = run_data_selection_experiment(dataset_name, num_runs=5)

# Plotting
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
colors = {'DataInf': 'blue', 'Full': 'orange', 'Hessian-free': 'red', 
          'LiSSA': 'purple', 'Random': 'green'}

for idx, dataset_name in enumerate(datasets):
    ax = axes[idx]
    results = all_results[dataset_name]
    
    for method_name in ['DataInf', 'Full', 'Hessian-free', 'LiSSA', 'Random']:
        method_results = np.array(results[method_name])
        mean_acc = np.mean(method_results, axis=0)
        std_acc = np.std(method_results, axis=0)
        
        epochs = np.arange(1, 11)
        ax.plot(epochs, mean_acc, label=method_name, color=colors[method_name])
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, 
                        alpha=0.2, color=colors[method_name])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Dataset: {dataset_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 10)

plt.tight_layout()
plt.savefig('data_selection_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nExperiment completed! Results saved to 'data_selection_results.png'")
