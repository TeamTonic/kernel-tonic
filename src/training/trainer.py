"""
Training script for Kernel Tonic model with custom kernels, FP8 training, and Colossal OSCAR 1.0.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
from typing import Optional, Dict, Any
from tqdm import tqdm
import wandb

from ..model.config import KernelTonicConfig, get_small_config, get_medium_config, get_large_config
from ..model.architecture import KernelTonicForCausalLM
from ..model.kernel_integration import quantize_model_fp8, dequantize_model_fp8
from .datasets import get_oscar_dataset


class OSCARDataset(IterableDataset):
    """Iterable dataset wrapper for Colossal OSCAR 1.0."""
    
    def __init__(self, dataset, tokenizer, max_length: int = 2048, buffer_size: int = 1000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Split dataset across workers
        if worker_info is not None:
            per_worker = len(self.dataset) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker
            dataset = self.dataset.select(range(start, end))
        else:
            dataset = self.dataset
        
        buffer = []
        for sample in dataset:
            # Tokenize the text
            text = sample.get('content', '')
            if not text:
                continue
            
            # Tokenize and truncate
            tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            if len(tokens) < 10:  # Skip very short sequences
                continue
            
            buffer.append(tokens)
            
            # Yield when buffer is full
            if len(buffer) >= self.buffer_size:
                for item in buffer:
                    yield torch.tensor(item, dtype=torch.long)
                buffer = []
        
        # Yield remaining items
        for item in buffer:
            yield torch.tensor(item, dtype=torch.long)


class FP8Trainer:
    """Trainer for FP8-native training with custom kernels."""
    
    def __init__(self, config: KernelTonicConfig, model: nn.Module, 
                 tokenizer, device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize optimizer with FP8 support
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-4,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_training_steps if hasattr(config, 'num_training_steps') else 100000
        )
        
        # Quantize model to FP8
        self.model = quantize_model_fp8(self.model, config)
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir='./logs')
        wandb.init(project="kernel-tonic", config=config.to_dict())
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step with FP8 precision."""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Prepare inputs
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def train(self, train_dataloader: DataLoader, num_epochs: int = 1):
        """Main training loop."""
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics['learning_rate']:.2e}"
                })
                
                # Log metrics
                self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
                self.writer.add_scalar('train/learning_rate', metrics['learning_rate'], self.global_step)
                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/learning_rate': metrics['learning_rate'],
                    'global_step': self.global_step
                })
                
                self.global_step += 1
                
                # Save checkpoint
                if self.global_step % 1000 == 0:
                    self.save_checkpoint()
            
            # Epoch summary
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(is_best=True)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f'checkpoint_step_{self.global_step}.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'best_model.pt')
            print(f"New best model saved with loss: {self.best_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint from step {self.global_step}")


def create_tokenizer():
    """Create a simple tokenizer for training."""
    from transformers import AutoTokenizer
    
    # Use a base tokenizer (you can replace with your preferred one)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train Kernel Tonic model with custom kernels')
    parser.add_argument('--config', type=str, default='small', 
                       choices=['small', 'medium', 'large', 'xlarge'],
                       help='Model configuration')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check for HF token
    if not os.environ.get('HF_TOKEN'):
        raise RuntimeError("Please set the HF_TOKEN environment variable")
    
    # Create configuration
    if args.config == 'small':
        config = get_small_config()
    elif args.config == 'medium':
        config = get_medium_config()
    elif args.config == 'large':
        config = get_large_config()
    else:
        config = get_xlarge_config()
    
    # Add training-specific config
    config.learning_rate = 1e-4
    config.num_training_steps = 100000
    
    # Create model
    model = KernelTonicForCausalLM(config)
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Create trainer
    trainer = FP8Trainer(config, model, tokenizer, device=args.device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Load dataset
    logger.info("Loading Colossal OSCAR 1.0 dataset...")
    dataset = get_oscar_dataset()
    
    # Create dataloader
    train_dataset = OSCARDataset(dataset, tokenizer, max_length=args.max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train(train_dataloader, num_epochs=args.num_epochs)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 