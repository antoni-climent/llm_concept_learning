#!/usr/bin/env python3
"""
Script to find all metrics_summary.csv files in subdirectories and create plots for each.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def create_metrics_plot(csv_path):
    """
    Create a plot from a metrics_summary.csv file and save it in the same directory.
    
    Args:
        csv_path: Path to the metrics_summary.csv file
    """
    csv_name = Path(csv_path).name
    # Determine type and output name
    if 'train' in csv_name:
        tag = "Train"
        out_name = 'metrics_train_plot.png'
    elif 'val' in csv_name:
        tag = "Val"
        out_name = 'metrics_val_plot.png'
    else:
        tag = ""
        out_name = 'metrics_plot.png'

    # Check if plot already exists
    output_path = Path(csv_path).parent / out_name
    if output_path.exists():
        print(f"⊙ Plot already exists: {output_path}")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f" ({tag})" if tag else ""
    fig.suptitle(f'Metrics Summary{title_suffix} - {Path(csv_path).parent.name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy over steps
    axes[0, 0].plot(df['step'], df['accuracy'], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    axes[0, 0].set_xlabel('Step', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_title('Accuracy over Training Steps', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: TP, FP, FN, TN over steps
    axes[0, 1].plot(df['step'], df['TP'], marker='o', label='True Positive', linewidth=2, markersize=5)
    axes[0, 1].plot(df['step'], df['FP'], marker='s', label='False Positive', linewidth=2, markersize=5)
    axes[0, 1].plot(df['step'], df['FN'], marker='^', label='False Negative', linewidth=2, markersize=5)
    axes[0, 1].plot(df['step'], df['TN'], marker='d', label='True Negative', linewidth=2, markersize=5)
    axes[0, 1].set_xlabel('Step', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Confusion Matrix Components', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Stacked bar chart of confusion matrix
    width = 0.6
    x_pos = range(len(df['step']))
    axes[1, 0].bar(x_pos, df['TP'], width, label='True Positive', color='#06A77D')
    axes[1, 0].bar(x_pos, df['FP'], width, bottom=df['TP'], label='False Positive', color='#F77F00')
    axes[1, 0].bar(x_pos, df['FN'], width, bottom=df['TP']+df['FP'], label='False Negative', color='#D62828')
    axes[1, 0].bar(x_pos, df['TN'], width, bottom=df['TP']+df['FP']+df['FN'], label='True Negative', color='#003049')
    if 'UNKNOWN' in df.columns:
        axes[1, 0].bar(x_pos, df['UNKNOWN'], width, bottom=df['TP']+df['FP']+df['FN']+df['TN'], 
                       label='Unknown', color='#CCCCCC')
    axes[1, 0].set_xlabel('Step', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Stacked Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(df['step'], rotation=45)
    axes[1, 0].legend(loc='upper left', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: UNKNOWN count over steps (if present)
    if 'UNKNOWN' in df.columns:
        axes[1, 1].plot(df['step'], df['UNKNOWN'], marker='o', linewidth=2, markersize=6, color='#A4133C')
        axes[1, 1].set_xlabel('Step', fontsize=11)
        axes[1, 1].set_ylabel('Unknown Count', fontsize=11)
        axes[1, 1].set_title('Unknown Predictions over Steps', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no UNKNOWN column, show precision and recall
        precision = df['TP'] / (df['TP'] + df['FP'])
        recall = df['TP'] / (df['TP'] + df['FN'])
        axes[1, 1].plot(df['step'], precision, marker='o', label='Precision', linewidth=2, markersize=6)
        axes[1, 1].plot(df['step'], recall, marker='s', label='Recall', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Step', fontsize=11)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].set_title('Precision and Recall', fontsize=12, fontweight='bold')
        axes[1, 1].legend(loc='best', fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot in the same directory as the CSV file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created plot: {output_path}")


def main():
    """
    Find all metrics_summary.csv files in subdirectories and create plots for each.
    """
    current_dir = Path.cwd()
    # Find all metrics summary files in subdirectories
    # Supports metrics_summary.csv, metrics_train_summary.csv, metrics_val_summary.csv
    csv_files = []
    for pattern in ['*/metrics_summary.csv', '*/metrics_train_summary.csv', '*/metrics_val_summary.csv']:
        csv_files.extend(list(current_dir.glob(pattern)))
    
    # Sort for consistent output
    csv_files.sort()
    
    if not csv_files:
        print("No metrics summary files found in subdirectories.")
        return
    
    print(f"Found {len(csv_files)} metrics summary file(s)\n")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            print(f"Processing: {csv_file.parent.name}/{csv_file.name}")
            create_metrics_plot(csv_file)
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
    
    print("-" * 80)
    print(f"Done! Processed {len(csv_files)} file(s).")


if __name__ == "__main__":
    main()
