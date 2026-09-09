"""Model-training utilities."""
import os
import time
import math
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import sys
import random
import numpy as np
from shilab_classifier.models.model_definitions import create_model
from shilab_classifier.training.preprocessing import (
    pad_to_square_transform,
    pad_to_square_299_transform,
    calculate_mean_std
)

def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_lr_schedule(optimizer, epoch, num_epochs, initial_lr):
    """Cosine lr schedule."""
    lr = initial_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs,
                initial_lr,
                patience,
                device,
                log_file=None
                ):
    """Train one model instance."""
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_model_wts = None
    best_detail_matrix = None
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_malignant_acc': [], 'train_benign_acc': [],
        'val_malignant_acc': [], 'val_benign_acc': [],
        'learning_rate': []
    }
    
    patience = patience
    counter = 0
    best_val_loss = float('inf')
    
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        log_message(f"Epoch {epoch+1}/{num_epochs} start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        current_lr = cosine_lr_schedule(optimizer, epoch, num_epochs, initial_lr)
        log_message(f"Current learning rate: {current_lr:.8f}")
        history['learning_rate'].append(current_lr)
        
        model.train()
        log_message("Starting training phase...")
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        try:
            log_message(f"Number of training batches: {len(train_loader)}")
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple) and len(outputs) > 1:
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
            log_message("Training phase complete!")
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
        except Exception as e:
            log_message(f"Error in training phase: {e}")
            traceback.print_exc()
            
            if log_file:
                error_file = log_file.replace('.log', '_error.log')
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"Error in training phase: {e}\n")
                    f.write(traceback.format_exc())
            
            return model, history, 0, 0, {}
        
        model.eval()
        log_message("Starting validation phase...")
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        try:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    if inputs.shape[0] == 1:
                        inputs = torch.cat([inputs, inputs], dim=0)
                        labels = torch.cat([labels, labels], dim=0)
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple) and len(outputs) > 1:
                        logits = outputs[0]
                    else:
                        logits = outputs
                    loss = criterion(logits, labels)
                        
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            log_message("Validation phase complete!")
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
        except Exception as e:
            log_message(f"Error in validation phase: {e}")
            traceback.print_exc()
            
            if log_file:
                error_file = log_file.replace('.log', '_error.log')
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"Error in validation phase: {e}\n")
                    f.write(traceback.format_exc())
                    
            return model, history, 0, 0, {}
        
        try:
            train_cm = confusion_matrix(train_labels, train_preds)
            val_cm = confusion_matrix(val_labels, val_preds)
            
            detail_matrix = {
                'train_malignant_correct': train_cm[1, 1] if train_cm.shape == (2, 2) else 0,
                'train_malignant_total': train_cm[1, :].sum() if train_cm.shape == (2, 2) else 0,
                'train_benign_correct': train_cm[0, 0] if train_cm.shape == (2, 2) else 0,
                'train_benign_total': train_cm[0, :].sum() if train_cm.shape == (2, 2) else 0,
                'train_malignant_acc': (train_cm[1, 1] / train_cm[1, :].sum()) if train_cm.shape == (2, 2) and train_cm[1, :].sum() > 0 else 0.0,
                'train_benign_acc': (train_cm[0, 0] / train_cm[0, :].sum()) if train_cm.shape == (2, 2) and train_cm[0, :].sum() > 0 else 0.0,

                'val_malignant_correct': val_cm[1, 1] if val_cm.shape == (2, 2) else 0,
                'val_malignant_total': val_cm[1, :].sum() if val_cm.shape == (2, 2) else 0,
                'val_benign_correct': val_cm[0, 0] if val_cm.shape == (2, 2) else 0,
                'val_benign_total': val_cm[0, :].sum() if val_cm.shape == (2, 2) else 0,
                'val_malignant_acc': (val_cm[1, 1] / val_cm[1, :].sum()) if val_cm.shape == (2, 2) and val_cm[1, :].sum() > 0 else 0.0,
                'val_benign_acc': (val_cm[0, 0] / val_cm[0, :].sum()) if val_cm.shape == (2, 2) and val_cm[0, :].sum() > 0 else 0.0
            }
        except Exception as e:
            log_message(f"Error computing confusion matrix: {e}")
            traceback.print_exc()
            
            if log_file:
                error_file = log_file.replace('.log', '_error.log')
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"Error computing confusion matrix: {e}\n")
                    f.write(traceback.format_exc())
                    
            detail_matrix = {
                'train_malignant_acc': 0.0, 'train_benign_acc': 0.0,
                'val_malignant_acc': 0.0, 'val_benign_acc': 0.0
            }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_model_wts = model.state_dict().copy()
            best_detail_matrix = detail_matrix.copy()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_malignant_acc'].append(detail_matrix['train_malignant_acc'])
        history['train_benign_acc'].append(detail_matrix['train_benign_acc'])
        history['val_malignant_acc'].append(detail_matrix['val_malignant_acc'])
        history['val_benign_acc'].append(detail_matrix['val_benign_acc'])
        
        epoch_time = time.time() - start_time
        log_message(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'LR: {current_lr:.8f} | Time: {epoch_time:.2f}s')
    
        if counter >= patience:
            log_message(f'Early stopping triggered after {epoch+1} epochs (validation loss not improved for {patience} epochs)')
            break
    
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, history, best_val_acc, best_train_acc, best_detail_matrix


def plot_training_curves(history, model_name, fold, save_dir):
    """Plot training curves."""
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Fold {fold} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower()}_fold_{fold}_loss.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} - Fold {fold} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower()}_fold_{fold}_accuracy.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    train_balanced_acc = [(b + m) / 2 for b, m in zip(history['train_benign_acc'], history['train_malignant_acc'])]
    val_balanced_acc = [(b + m) / 2 for b, m in zip(history['val_benign_acc'], history['val_malignant_acc'])]
    
    plt.plot(train_balanced_acc, label='Train Balanced Acc')
    plt.plot(val_balanced_acc, label='Val Balanced Acc')
    plt.title(f'{model_name} - Fold {fold} Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower()}_fold_{fold}_balanced_acc.png'))
    plt.close()


def train_single_model_fold(   
    model_name, 
    fold, 
    train_dir, 
    val_dir, 
    save_dir, 
    num_epochs=50, 
    batch_size=16, 
    initial_lr=0.001, 
    device='cuda',
    num_classes=2,
    num_workers=0,
    patience=10,
    seed=42):
    """Train one model for one cross-validation fold."""
    set_seed(seed)
    
    log_file = os.path.join(save_dir, f'{model_name.lower()}_fold_{fold}.log')
    
    def log_message(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    log_message(f"\n{'='*60}")
    log_message(f"Training {model_name} - Fold {fold}")
    log_message(f"Random seed: {seed}")
    log_message(f"{'='*60}")
    
    log_message("Computing mean and std of training set...")
    mean, std = calculate_mean_std(train_dir, model_name)
    log_message(f"Mean: {mean}")
    log_message(f"Std: {std}")
    
    if 'Inception' in model_name:
        pad_transform = pad_to_square_299_transform
    else:
        pad_transform = pad_to_square_transform
    
    train_transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Lambda(pad_transform),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    log_message("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    generator = torch.Generator()
    generator.manual_seed(seed) 
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True,
        generator=generator, 
        worker_init_fn=seed_worker
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker
    )
    
    log_message(f"Training samples: {len(train_dataset)}")
    log_message(f"Validation samples: {len(val_dataset)}")
    log_message(f"Classes: {train_dataset.classes}")
    
    log_message(f"Creating model: {model_name}")
    model = create_model(model_name, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)
    
    log_message(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    
    model, history, best_val_acc, best_train_acc, best_detail_matrix = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs, initial_lr, patience, device, log_file
    )
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    model_save_path = os.path.join(save_dir, f'{model_name.lower()}_fold_{fold}.pth')
    torch.save(model.state_dict(), model_save_path)
    log_message(f"✅ Model saved to: {model_save_path}")
    
    plot_training_curves(history, model_name, fold, save_dir)
    log_message(f"✅ Training curves saved to: {os.path.join(save_dir, 'plots')}")
    
    log_message(f"\n{'='*60}")
    log_message(f"{model_name} - Fold {fold} training complete!")
    log_message(f"Best validation accuracy: {best_val_acc:.4f}")
    log_message(f"Corresponding training accuracy: {best_train_acc:.4f}")
    if best_detail_matrix:
        log_message(f"Detailed metrics:")
        log_message(f"  Train:")
        log_message(f"    Benign  - correct: {best_detail_matrix.get('train_benign_correct', 0)}/{best_detail_matrix.get('train_benign_total', 0)}, "
              f"acc: {best_detail_matrix.get('train_benign_acc', 0):.4f}")
        log_message(f"    Malignant - correct: {best_detail_matrix.get('train_malignant_correct', 0)}/{best_detail_matrix.get('train_malignant_total', 0)}, "
              f"acc: {best_detail_matrix.get('train_malignant_acc', 0):.4f}")
        log_message(f"  Validation:")
        log_message(f"    Benign  - correct: {best_detail_matrix.get('val_benign_correct', 0)}/{best_detail_matrix.get('val_benign_total', 0)}, "
              f"acc: {best_detail_matrix.get('val_benign_acc', 0):.4f}")
        log_message(f"    Malignant - correct: {best_detail_matrix.get('val_malignant_correct', 0)}/{best_detail_matrix.get('val_malignant_total', 0)}, "
              f"acc: {best_detail_matrix.get('val_malignant_acc', 0):.4f}")
    log_message(f"Training time: {training_time:.2f} minutes")
    log_message(f"{'='*60}")
    
    return history


def train_all_models(model_list, cv_data_dir, output_dir, num_folds, 
                     num_epochs, batch_size, initial_lr, device, num_classes, 
                     num_workers, patience, save_log=True,
                     seed=42):
    """Train all requested models across cross-validation folds."""
    print(f"\n{'='*60}")
    print(f"Starting training for all models")
    print(f"Number of models: {len(model_list)}")
    print(f"Number of folds: {num_folds}")
    print(f"Total training runs: {len(model_list) * num_folds}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}")
    
    main_log_file = os.path.join(output_dir, "training_main.log")
    with open(main_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of models: {len(model_list)}\n")
        f.write(f"Number of folds: {num_folds}\n")
        f.write(f"Total training runs: {len(model_list) * num_folds}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"{'='*60}\n")
    
    total_start_time = time.time()
    training_logs = []
    
    for model_name in model_list:
        model_save_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        for fold in range(1, num_folds + 1):
            train_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'train')
            val_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'val')
            
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                error_msg = f"❌ Error: data path not found - {train_dir} or {val_dir}"
                print(error_msg)
                with open(main_log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + '\n')
                continue
            
            try:
                fold_start_time = time.time()
                history = train_single_model_fold(
                    model_name=model_name,
                    fold=fold,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    save_dir=model_save_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    initial_lr=initial_lr,
                    device=device,
                    num_classes=num_classes,
                    num_workers=num_workers,
                    patience=patience,
                    seed=seed 
                )
                fold_time = (time.time() - fold_start_time) / 60
                
                log_entry = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Model': model_name,
                    'Fold': fold,
                    'Seed': seed, 
                    'Best Val Acc': max(history['val_acc']) if history['val_acc'] else 0.0,
                    'Corresponding Train Acc': history['train_acc'][history['val_acc'].index(max(history['val_acc']))] if history['val_acc'] else 0.0,
                    'Training Time (min)': f"{fold_time:.2f}"
                }
                training_logs.append(log_entry)
                
                with open(main_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Completed {model_name} - Fold {fold} | "
                           f"Best Val Acc: {log_entry['Best Val Acc']:.4f} | "
                           f"Training Time: {log_entry['Training Time (min)']} min\n")
                
            except Exception as e:
                error_msg = f"❌ Error training {model_name} Fold {fold}: {e}"
                print(error_msg)
                traceback.print_exc()
                
                with open(main_log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + '\n')
                    f.write(traceback.format_exc() + '\n')
                
                continue
    
    total_time = (time.time() - total_start_time) / 60
    
    completion_msg = f"\n{'='*60}\n✅ All models trained successfully!\nTotal training time: {total_time:.2f} minutes\n{'='*60}"
    print(completion_msg)
    with open(main_log_file, 'a', encoding='utf-8') as f:
        f.write(completion_msg + '\n')
    
    if save_log and training_logs:
        log_path = os.path.join(output_dir, "training_log.xlsx")
        df = pd.DataFrame(training_logs)
        df.to_excel(log_path, index=False, engine='openpyxl')
        print(f"\n📝 Training log saved to: {log_path}")
    
    return {
        'total_time': total_time,
        'training_logs': training_logs,
        'output_dir': output_dir
    }