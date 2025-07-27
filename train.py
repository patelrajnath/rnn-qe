import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import AttentionRNNQE, BaselineRNNQE
from preprocessing import create_dataloaders


class Trainer:
    """Trainer class for RNN-based Quality Estimation"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            src_tokens = batch['src_tokens'].to(self.device)
            tgt_tokens = batch['tgt_tokens'].to(self.device)
            src_lengths = batch['src_lengths'].to(self.device)
            tgt_lengths = batch['tgt_lengths'].to(self.device)
            quality_scores = batch['quality_scores'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(src_tokens, tgt_tokens, src_lengths, tgt_lengths)
            
            # Compute loss
            # For token-level QE, expand quality score to match token sequence
            if len(predictions.shape) == 2:  # [batch, tgt_len]
                # Use mean quality score for all tokens (sentence-level QE)
                target_scores = quality_scores.unsqueeze(1).expand_as(predictions)
                loss = self.mse_loss(predictions, target_scores)
            else:  # [batch, 1]
                loss = self.mse_loss(predictions.squeeze(), quality_scores)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                src_tokens = batch['src_tokens'].to(self.device)
                tgt_tokens = batch['tgt_tokens'].to(self.device)
                src_lengths = batch['src_lengths'].to(self.device)
                tgt_lengths = batch['tgt_lengths'].to(self.device)
                quality_scores = batch['quality_scores'].to(self.device)
                
                # Forward pass
                predictions = self.model(src_tokens, tgt_tokens, src_lengths, tgt_lengths)
                
                # Compute loss
                if len(predictions.shape) == 2:
                    # Take mean prediction across tokens for sentence-level evaluation
                    predictions_mean = predictions.mean(dim=1)
                    loss = self.mse_loss(predictions_mean, quality_scores)
                    all_predictions.extend(predictions_mean.cpu().numpy())
                else:
                    loss = self.mse_loss(predictions.squeeze(), quality_scores)
                    all_predictions.extend(predictions.squeeze().cpu().numpy())
                
                all_targets.extend(quality_scores.cpu().numpy())
                total_loss += loss.item()
        
        # Compute metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        pearson_corr = np.corrcoef(all_targets, all_predictions)[0, 1]
        r2 = r2_score(all_targets, all_predictions)
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'mae': mae,
            'rmse': rmse,
            'pearson_corr': pearson_corr,
            'r2': r2
        }
        
        return metrics
    
    def train(self, num_epochs=50, save_dir='checkpoints'):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Logging
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Tensorboard logging
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/MAE', val_metrics['mae'], epoch)
            writer.add_scalar('Metrics/RMSE', val_metrics['rmse'], epoch)
            writer.add_scalar('Metrics/Pearson', val_metrics['pearson_corr'], epoch)
            writer.add_scalar('Metrics/R2', val_metrics['r2'], epoch)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  MAE: {val_metrics['mae']:.4f}")
            print(f"  RMSE: {val_metrics['rmse']:.4f}")
            print(f"  Pearson: {val_metrics['pearson_corr']:.4f}")
            print(f"  R2: {val_metrics['r2']:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'train_loss': train_loss
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_metrics': self.val_metrics
                }, checkpoint_path)
        
        writer.close()
        return self.train_losses, self.val_losses, self.val_metrics
    
    def plot_training_curves(self, save_dir='checkpoints'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        maes = [m['mae'] for m in self.val_metrics]
        axes[0, 1].plot(maes, label='MAE', color='red')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Pearson correlation
        pearsons = [m['pearson_corr'] for m in self.val_metrics]
        axes[1, 0].plot(pearsons, label='Pearson Correlation', color='green')
        axes[1, 0].set_title('Pearson Correlation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Pearson r')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # R2 score
        r2_scores = [m['r2'] for m in self.val_metrics]
        axes[1, 1].plot(r2_scores, label='R² Score', color='purple')
        axes[1, 1].set_title('R² Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            src_tokens = batch['src_tokens'].to(device)
            tgt_tokens = batch['tgt_tokens'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            tgt_lengths = batch['tgt_lengths'].to(device)
            quality_scores = batch['quality_scores'].to(device)
            
            predictions = model(src_tokens, tgt_tokens, src_lengths, tgt_lengths)
            
            if len(predictions.shape) == 2:
                predictions = predictions.mean(dim=1)
            
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(quality_scores.cpu().numpy())
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    pearson_corr = np.corrcoef(all_targets, all_predictions)[0, 1]
    r2 = r2_score(all_targets, all_predictions)
    
    results = {
        'test_mae': mae,
        'test_rmse': rmse,
        'test_pearson': pearson_corr,
        'test_r2': r2,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return results


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RNN Quality Estimation Model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--use_synthetic', action='store_true')
    parser.add_argument('--synthetic_samples', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_type', type=str, choices=['attention', 'baseline'], default='attention')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"Using device: {args.device}")
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_synthetic=args.use_synthetic,
        synthetic_samples=args.synthetic_samples
    )
    
    # Create model
    src_vocab_size = len(data_loaders['src_vocab'])
    tgt_vocab_size = len(data_loaders['tgt_vocab'])
    
    if args.model_type == 'attention':
        model = AttentionRNNQE(
            vocab_size_src=src_vocab_size,
            vocab_size_tgt=tgt_vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = BaselineRNNQE(
            vocab_size_src=src_vocab_size,
            vocab_size_tgt=tgt_vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim
        )
    
    print(f"Created {args.model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train_loader'],
        val_loader=data_loaders['val_loader'],
        device=args.device
    )
    
    # Train model
    train_losses, val_losses, val_metrics = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Plot training curves
    trainer.plot_training_curves(save_dir=args.save_dir)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        test_loader=data_loaders['test_loader'],
        device=args.device
    )
    
    # Save results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"MAE: {test_results['test_mae']:.4f}")
    print(f"RMSE: {test_results['test_rmse']:.4f}")
    print(f"Pearson Correlation: {test_results['test_pearson']:.4f}")
    print(f"R² Score: {test_results['test_r2']:.4f}")
    print("="*50)
    
    # Save model info
    model_info = {
        'model_type': args.model_type,
        'parameters': sum(p.numel() for p in model.parameters()),
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'best_val_loss': min(val_losses) if val_losses else None,
        'final_test_results': test_results
    }
    
    with open(os.path.join(args.save_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)


if __name__ == "__main__":
    main()