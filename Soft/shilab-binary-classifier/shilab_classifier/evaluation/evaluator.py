"""Model-evaluation utilities."""

import os
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from datetime import datetime

from shilab_classifier.models.model_definitions import create_model
from shilab_classifier.training.preprocessing import (
    pad_to_square_transform,
    pad_to_square_299_transform,
    calculate_mean_std
)
from shilab_classifier.evaluation.metrics import calculate_metrics, print_metrics


def predict_and_save_probabilities(model, dataloader, dataset_name, save_dir, device='cuda'):
    """Predict and save probabilities."""
    model.eval()

    all_logits = []
    all_probs  = []
    all_preds  = []
    all_labels = []
    all_paths  = []

    print(f"Predicting probabilities for {dataset_name}...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")

            inputs  = inputs.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())

            if hasattr(dataloader.dataset, 'samples'):
                batch_paths = [dataloader.dataset.samples[i][0] for i in range(
                    batch_idx * dataloader.batch_size,
                    min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                )]
                all_paths.extend(batch_paths)

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    results_df = pd.DataFrame({
        'image_path':       all_paths if all_paths else [''] * len(all_labels),
        'filename':         [os.path.basename(p) if p else '' for p in all_paths] if all_paths else [''] * len(all_labels),
        'logit_benign':     all_logits[:, 0],
        'logit_malignant':  all_logits[:, 1],
        'prob_benign':      all_probs[:, 0],
        'prob_malignant':   all_probs[:, 1],
        'predicted_class':  all_preds,
        'true_label':       all_labels
    })

    prob_save_path = os.path.join(save_dir, f'{dataset_name}_probabilities.csv')
    results_df.to_csv(prob_save_path, index=False)
    print(f"Probabilities saved to {prob_save_path}")

    metrics = calculate_metrics(all_labels, all_preds, all_probs[:, 1])

    metrics_df = pd.DataFrame([{
        'dataset':              dataset_name,
        'accuracy':             metrics['accuracy'],
        'balanced_accuracy':    metrics['balanced_accuracy'],
        'sensitivity':          metrics['sensitivity'],
        'specificity':          metrics['specificity'],
        'precision_benign':     metrics['precision_benign'],
        'recall_benign':        metrics['recall_benign'],
        'f1_benign':            metrics['f1_benign'],
        'precision_malignant':  metrics['precision_malignant'],
        'recall_malignant':     metrics['recall_malignant'],
        'f1_malignant':         metrics['f1_malignant'],
        'mcc':                  metrics['mcc'],
        'auc':                  metrics['auc'],
        'benign_correct':       metrics['benign_correct'],
        'benign_total':         metrics['benign_total'],
        'benign_acc':           metrics['benign_acc'],
        'malignant_correct':    metrics['malignant_correct'],
        'malignant_total':      metrics['malignant_total'],
        'malignant_acc':        metrics['malignant_acc']
    }])

    metrics_save_path = os.path.join(save_dir, f'{dataset_name}_metrics.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Metrics saved to {metrics_save_path}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_save_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_save_path}")

    return metrics


def plot_roc_curve(train_probs, train_labels, val_probs, val_labels,
                   test_probs, test_labels, save_path):
    """Plot ROC curves for train, val, and test sets."""
    plt.figure(figsize=(10, 8))

    datasets_info = [
        ('train', train_probs, train_labels, 'blue'),
        ('val',   val_probs,   val_labels,   'green'),
        ('test',  test_probs,  test_labels,  'red')
    ]

    for name, probs, labels, color in datasets_info:
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def evaluate_single_model(model_name, cv_data_dir, test_dir, models_dir,
                           output_dir, num_folds=5, batch_size=16,
                           num_workers=0, num_classes=2, device='cuda'):
    """Evaluate one trained model."""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}")

    model_results_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    roc_dir = os.path.join(model_results_dir, 'roc_curves')
    os.makedirs(roc_dir, exist_ok=True)

    all_results = []

    for fold in range(1, num_folds + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")

        fold_results_dir = os.path.join(model_results_dir, f'fold_{fold}')
        os.makedirs(fold_results_dir, exist_ok=True)

        train_dir = os.path.join(cv_data_dir, f'fold_{fold}', 'train')
        val_dir   = os.path.join(cv_data_dir, f'fold_{fold}', 'val')

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"Error: data directory not found, skipping fold {fold}")
            continue

        print("Computing mean and std...")
        mean, std = calculate_mean_std(train_dir, model_name)
        print(f"Mean: {mean}, Std: {std}")

        pad_transform = (
            pad_to_square_299_transform if 'Inception' in model_name
            else pad_to_square_transform
        )

        transform = transforms.Compose([
            transforms.Lambda(pad_transform),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print("Loading datasets...")
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        val_dataset   = datasets.ImageFolder(val_dir,   transform=transform)
        test_dataset  = datasets.ImageFolder(test_dir,  transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)

        model_dir = os.path.join(models_dir, model_name)

        possible_names = [
            f'{model_name.lower()}_fold_{fold}.pth',
            f'{model_name}_fold_{fold}.pth',
            f'{model_name.lower()}_fold{fold}.pth',
            f'{model_name}_fold{fold}.pth',
            f'fold_{fold}.pth',
            f'fold{fold}.pth',
            f'model_fold_{fold}.pth'
        ]

        model_path = None
        for name in possible_names:
            temp_path = os.path.join(model_dir, name)
            if os.path.exists(temp_path):
                model_path = temp_path
                break

        if model_path is None:
            print(f"Error: model file for fold {fold} not found, skipping")
            continue

        print(f"Loading model weights: {model_path}")
        model = create_model(model_name, num_classes=num_classes)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            traceback.print_exc()
            continue

        train_metrics = predict_and_save_probabilities(
            model, train_loader, 'train', fold_results_dir, device)
        val_metrics = predict_and_save_probabilities(
            model, val_loader, 'val', fold_results_dir, device)
        test_metrics = predict_and_save_probabilities(
            model, test_loader, 'test', fold_results_dir, device)

        print(f"\n{model_name} Fold {fold} results:")
        print_metrics(train_metrics, "Train")
        print_metrics(val_metrics,   "Val")
        print_metrics(test_metrics,  "Test")

        train_probs_df = pd.read_csv(os.path.join(fold_results_dir, 'train_probabilities.csv'))
        val_probs_df   = pd.read_csv(os.path.join(fold_results_dir, 'val_probabilities.csv'))
        test_probs_df  = pd.read_csv(os.path.join(fold_results_dir, 'test_probabilities.csv'))

        roc_save_path = os.path.join(roc_dir, f'roc_curve_fold_{fold}.png')
        plot_roc_curve(
            train_probs_df['prob_malignant'].values, train_probs_df['true_label'].values,
            val_probs_df['prob_malignant'].values,   val_probs_df['true_label'].values,
            test_probs_df['prob_malignant'].values,  test_probs_df['true_label'].values,
            roc_save_path
        )

        fold_result = {
            'model': model_name,
            'fold':  fold,
            'train_accuracy':             train_metrics['accuracy'],
            'train_balanced_accuracy':    train_metrics['balanced_accuracy'],
            'train_sensitivity':          train_metrics['sensitivity'],
            'train_specificity':          train_metrics['specificity'],
            'train_precision_benign':     train_metrics['precision_benign'],
            'train_recall_benign':        train_metrics['recall_benign'],
            'train_f1_benign':            train_metrics['f1_benign'],
            'train_precision_malignant':  train_metrics['precision_malignant'],
            'train_recall_malignant':     train_metrics['recall_malignant'],
            'train_f1_malignant':         train_metrics['f1_malignant'],
            'train_mcc':                  train_metrics['mcc'],
            'train_auc':                  train_metrics['auc'],
            'train_benign_acc':           train_metrics['benign_acc'],
            'train_malignant_acc':        train_metrics['malignant_acc'],
            'val_accuracy':               val_metrics['accuracy'],
            'val_balanced_accuracy':      val_metrics['balanced_accuracy'],
            'val_sensitivity':            val_metrics['sensitivity'],
            'val_specificity':            val_metrics['specificity'],
            'val_precision_benign':       val_metrics['precision_benign'],
            'val_recall_benign':          val_metrics['recall_benign'],
            'val_f1_benign':              val_metrics['f1_benign'],
            'val_precision_malignant':    val_metrics['precision_malignant'],
            'val_recall_malignant':       val_metrics['recall_malignant'],
            'val_f1_malignant':           val_metrics['f1_malignant'],
            'val_mcc':                    val_metrics['mcc'],
            'val_auc':                    val_metrics['auc'],
            'val_benign_acc':             val_metrics['benign_acc'],
            'val_malignant_acc':          val_metrics['malignant_acc'],
            'test_accuracy':              test_metrics['accuracy'],
            'test_balanced_accuracy':     test_metrics['balanced_accuracy'],
            'test_sensitivity':           test_metrics['sensitivity'],
            'test_specificity':           test_metrics['specificity'],
            'test_precision_benign':      test_metrics['precision_benign'],
            'test_recall_benign':         test_metrics['recall_benign'],
            'test_f1_benign':             test_metrics['f1_benign'],
            'test_precision_malignant':   test_metrics['precision_malignant'],
            'test_recall_malignant':      test_metrics['recall_malignant'],
            'test_f1_malignant':          test_metrics['f1_malignant'],
            'test_mcc':                   test_metrics['mcc'],
            'test_auc':                   test_metrics['auc'],
            'test_benign_acc':            test_metrics['benign_acc'],
            'test_malignant_acc':         test_metrics['malignant_acc']
        }

        all_results.append(fold_result)

    if not all_results:
        print(f"Warning: no folds were successfully evaluated for {model_name}")
        return None

    results_df  = pd.DataFrame(all_results)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    avg_result  = results_df[numeric_cols].mean().to_dict()
    avg_result['model'] = model_name
    avg_result['fold']  = 'Average'

    avg_df     = pd.DataFrame([avg_result])
    results_df = pd.concat([results_df, avg_df], ignore_index=True)

    detailed_results_path = os.path.join(model_results_dir, f'{model_name}_detailed_results.csv')
    results_df.to_csv(detailed_results_path, index=False)
    print(f"\nDetailed results saved to {detailed_results_path}")

    cv_results_path = os.path.join(model_results_dir, f'{model_name}_cross_validation_results.csv')
    avg_df.to_csv(cv_results_path, index=False)
    print(f"Cross-validation results saved to {cv_results_path}")

    return results_df


def evaluate_all_models(model_list, cv_data_dir, test_dir, models_dir,
                        output_dir, num_folds=5, batch_size=16, num_workers=0,
                        num_classes=2, device='cuda', save_summary=True):
    """Evaluate all requested trained models."""
    print(f"\n{'='*60}")
    print(f"Starting evaluation of all models")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    if model_list is None:
        if not os.path.exists(models_dir):
            raise ValueError(f"Models directory does not exist: {models_dir}")

        model_list = [d for d in os.listdir(models_dir)
                      if os.path.isdir(os.path.join(models_dir, d))]

        if not model_list:
            raise ValueError(f"No model folders found in {models_dir}")

    print(f"\nFound {len(model_list)} model(s):")
    for model_name in model_list:
        print(f"  - {model_name}")

    all_model_results = []
    start_time = time.time()

    for model_name in model_list:
        try:
            results_df = evaluate_single_model(
                model_name=model_name,
                cv_data_dir=cv_data_dir,
                test_dir=test_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                num_folds=num_folds,
                batch_size=batch_size,
                num_workers=num_workers,
                num_classes=num_classes,
                device=device
            )

            if results_df is not None:
                avg_result = results_df[results_df['fold'] == 'Average'].iloc[0].to_dict()
                all_model_results.append(avg_result)

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            traceback.print_exc()
            continue

    total_time = (time.time() - start_time) / 60

    if not all_model_results:
        print("Error: no models were successfully evaluated")
        return None

    if save_summary:
        _save_evaluation_summary(all_model_results, output_dir)

    print(f"\nTotal runtime: {total_time:.2f} minutes")

    return {
        'total_time': total_time,
        'num_models': len(all_model_results),
        'results':    all_model_results,
        'output_dir': output_dir
    }


def _save_evaluation_summary(all_model_results, output_dir):
    """Save evaluation summary."""
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    summary_df = pd.DataFrame(all_model_results)

    test_cols       = ['model'] + [col for col in summary_df.columns if col.startswith('test_')]
    summary_df_test = summary_df[test_cols].sort_values('test_mcc', ascending=False)

    summary_path = os.path.join(output_dir, 'all_models_summary.csv')
    summary_df_test.to_csv(summary_path, index=False)
    print(f"\nAll-model summary saved to {summary_path}")

    if has_tabulate:
        print("\nTest-set performance summary (all models):")
        print(tabulate(summary_df_test, headers='keys', tablefmt='psql',
                       showindex=False, floatfmt='.4f'))

    _plot_performance_heatmap(summary_df_test, output_dir)
    _analyze_top_models(summary_df, summary_df_test, output_dir, top_n=4)


def _plot_performance_heatmap(summary_df_test, output_dir):
    """Plot performance heatmap for all models on the test set."""
    plt.figure(figsize=(16, 10))

    metrics_to_plot = [
        'test_precision_benign',    'test_recall_benign',    'test_f1_benign',
        'test_precision_malignant', 'test_recall_malignant', 'test_f1_malignant',
        'test_balanced_accuracy',   'test_sensitivity',      'test_specificity',
        'test_mcc', 'test_auc'
    ]

    heatmap_data = summary_df_test[['model'] + metrics_to_plot].set_index('model')

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Model Performance Heatmap (Test Set)', fontsize=16, pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'all_models_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance heatmap saved to {heatmap_path}")


def _analyze_top_models(summary_df, summary_df_test, output_dir, top_n=4):
    """Analyze the top-N models ranked by test MCC."""
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    top_models      = summary_df_test.nlargest(top_n, 'test_mcc')
    top_model_names = top_models['model'].tolist()

    print(f"\nTop {top_n} models by MCC: {top_model_names}")

    best_folds_data = []

    for model_name in top_model_names:
        detailed_results_path = os.path.join(
            output_dir, model_name, f"{model_name}_detailed_results.csv"
        )

        if not os.path.exists(detailed_results_path):
            print(f"Warning: detailed results not found for {model_name}, skipping")
            continue

        detailed_df = pd.read_csv(detailed_results_path)
        fold_df     = detailed_df[detailed_df['fold'] != 'Average']

        if fold_df.empty:
            print(f"Warning: no fold data found for {model_name}")
            continue

        max_mcc_idx    = fold_df['test_mcc'].idxmax()
        best_fold      = fold_df.loc[max_mcc_idx, 'fold']
        best_mcc       = fold_df.loc[max_mcc_idx, 'test_mcc']
        best_folds_data.append(fold_df.loc[max_mcc_idx].to_dict())

        print(f"Model {model_name}: best fold={best_fold}, MCC={best_mcc:.4f}")

    if not best_folds_data:
        print("Warning: no best-fold data found for any model")
        return

    best_folds_df = pd.DataFrame(best_folds_data)

    key_metrics       = ['model', 'fold',
                         'train_sensitivity', 'train_specificity', 'train_auc',
                         'val_sensitivity',   'val_specificity',   'val_auc',
                         'test_sensitivity',  'test_specificity',  'test_auc']
    available_metrics = [m for m in key_metrics if m in best_folds_df.columns]
    top_summary       = best_folds_df[available_metrics]

    top_path = os.path.join(output_dir, f'top{top_n}_models_summary.csv')
    top_summary.to_csv(top_path, index=False)
    print(f"\nTop-{top_n} best-fold summary saved to {top_path}")

    if has_tabulate:
        print(f"\nTop-{top_n} models best-fold performance:")
        print(tabulate(top_summary, headers='keys', tablefmt='psql',
                       showindex=False, floatfmt='.4f'))

    metrics_to_plot       = [
        'test_precision_benign',    'test_recall_benign',    'test_f1_benign',
        'test_precision_malignant', 'test_recall_malignant', 'test_f1_malignant',
        'test_balanced_accuracy',   'test_sensitivity',      'test_specificity',
        'test_mcc', 'test_auc'
    ]
    available_plot_metrics = [m for m in metrics_to_plot if m in best_folds_df.columns]
    top_heatmap_data       = best_folds_df.set_index('model')[available_plot_metrics]

    plt.figure(figsize=(12, 6))
    sns.heatmap(top_heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title(f'Top {top_n} Models Best Fold Performance Heatmap (Test Set)', fontsize=14, pad=15)
    plt.xlabel('Metrics', fontsize=11)
    plt.ylabel('Models', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    top_heatmap_path = os.path.join(output_dir, f'top{top_n}_models_heatmap.png')
    plt.savefig(top_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top-{top_n} best-fold heatmap saved to {top_heatmap_path}")