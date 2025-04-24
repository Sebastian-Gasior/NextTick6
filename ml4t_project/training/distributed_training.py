"""
Distributed Training Framework für ML4T
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
from typing import Optional, List, Dict, Any
import os
from pathlib import Path
import json

class DistributedTrainer:
    """Verwaltet das verteilte Training über mehrere GPUs"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 num_gpus: int = torch.cuda.device_count(),
                 master_addr: str = 'localhost',
                 master_port: str = '12355'):
        
        self.model = model
        self.num_gpus = num_gpus
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Logging Setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Checkpointing
        self.checkpoint_dir = Path("ml4t_project/exports/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metriken
        self.metrics: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'gpu_utilization': []
        }
    
    def setup(self, rank: int, world_size: int):
        """Initialisiert die verteilte Umgebung"""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        
        # Initialisiere Prozessgruppe
        dist.init_process_group(
            backend='nccl',  # NCCL ist optimiert für GPU-Training
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Setze GPU für diesen Prozess
        torch.cuda.set_device(rank)
        
        self.logger.info(f"Prozess {rank} initialisiert auf GPU {rank}")
    
    def cleanup(self):
        """Beendet die verteilte Umgebung"""
        dist.destroy_process_group()
    
    def prepare_data(self, 
                    dataset: torch.utils.data.Dataset,
                    batch_size: int,
                    rank: int,
                    world_size: int) -> DataLoader:
        """Bereitet die Daten für verteiltes Training vor"""
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=sampler
        )
        
        return dataloader
    
    def train_model(self, 
                   rank: int,
                   world_size: int,
                   train_dataset: torch.utils.data.Dataset,
                   val_dataset: Optional[torch.utils.data.Dataset] = None,
                   epochs: int = 10,
                   batch_size: int = 32,
                   learning_rate: float = 0.001):
        """Führt das verteilte Training durch"""
        try:
            # Setup für diesen Prozess
            self.setup(rank, world_size)
            
            # Verschiebe Modell auf GPU
            model = self.model.to(rank)
            
            # Wrap mit DDP
            model = DDP(model, device_ids=[rank])
            
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Datenlader
            train_loader = self.prepare_data(train_dataset, batch_size, rank, world_size)
            if val_dataset:
                val_loader = self.prepare_data(val_dataset, batch_size, rank, world_size)
            
            # Training Loop
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(rank), target.to(rank)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.mse_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Log Fortschritt
                    if rank == 0 and batch_idx % 10 == 0:
                        self.logger.info(f'Epoch {epoch}: [{batch_idx}/{len(train_loader)}]'
                                       f' Loss: {loss.item():.6f}')
                
                # Validierung
                if val_dataset and rank == 0:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(rank), target.to(rank)
                            output = model(data)
                            val_loss += torch.nn.functional.mse_loss(output, target).item()
                    
                    val_loss /= len(val_loader)
                    self.metrics['val_loss'].append(val_loss)
                    self.logger.info(f'Validation Loss: {val_loss:.6f}')
                
                # Speichere Checkpoint
                if rank == 0:
                    self.save_checkpoint(model, optimizer, epoch)
                    
                # Synchronisiere Prozesse
                dist.barrier()
            
            self.cleanup()
            
        except Exception as e:
            self.logger.error(f"Fehler im Training auf Rank {rank}: {str(e)}")
            raise
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int):
        """Speichert den Trainingszustand"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': self.metrics
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Speichere Metriken separat für einfachen Zugriff
        metrics_path = self.checkpoint_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
        
        self.logger.info(f"Checkpoint gespeichert: {checkpoint_path}")
    
    def load_checkpoint(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Lädt einen gespeicherten Checkpoint"""
        if epoch is None:
            # Lade letzten Checkpoint
            checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if not checkpoints:
                raise FileNotFoundError("Keine Checkpoints gefunden")
            
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        self.logger.info(f"Checkpoint geladen: {checkpoint_path}")
        
        return checkpoint 