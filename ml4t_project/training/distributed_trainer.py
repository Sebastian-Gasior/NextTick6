import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """
    Ein Trainer für verteiltes Training mit PyTorch DDP.
    
    Attribute:
        model: Das zu trainierende PyTorch-Modell
        world_size: Anzahl der zu verwendenden Prozesse
        rank: Rang des aktuellen Prozesses
        device: Das zu verwendende Gerät (GPU/CPU)
    """
    
    def __init__(self, model, world_size=2, rank=0):
        """
        Initialisiert den verteilten Trainer.
        
        Args:
            model: PyTorch-Modell
            world_size: Anzahl der Prozesse
            rank: Rang des aktuellen Prozesses
        """
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialisiere die verteilte Umgebung
        self._setup_distributed()
        
        # Verschiebe das Modell auf das Gerät und wickle es in DDP
        self.model = self.model.to(self.device)
        self.model = DDP(self.model)
        
        # Initialisiere den Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.MSELoss()
        
        logger.info(f"Distributed Trainer initialisiert auf Rang {rank}")
    
    def _setup_distributed(self):
        """Initialisiert die verteilte Trainingsumgebung."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        try:
            dist.init_process_group(
                backend='gloo',
                rank=self.rank,
                world_size=self.world_size
            )
            logger.info(f"Prozessgruppe initialisiert für Rang {self.rank}")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der Prozessgruppe: {e}")
            raise
    
    def train_step(self, X, y):
        """
        Führt einen Trainingsschritt durch.
        
        Args:
            X: Eingabedaten (batch_size, seq_length, features)
            y: Zielwerte (batch_size, targets)
            
        Returns:
            float: Trainingsverlust
        """
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("Eingaben müssen PyTorch Tensoren sein")
            
        if len(X.shape) != 3:
            raise ValueError(f"X muss 3-dimensional sein (batch_size, seq_length, features), aber hat Form {X.shape}")
            
        if len(y.shape) != 2:
            raise ValueError(f"y muss 2-dimensional sein (batch_size, targets), aber hat Form {y.shape}")
            
        self.model.train()
        self.optimizer.zero_grad()
        
        # Verschiebe Daten auf das Gerät
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Forward Pass
        outputs, _ = self.model(X)  # Ignoriere den hidden state
        loss = self.criterion(outputs, y)
        
        # Backward Pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, X, y):
        """
        Führt einen Validierungsschritt durch.
        
        Args:
            X: Eingabedaten (batch_size, seq_length, features)
            y: Zielwerte (batch_size, targets)
            
        Returns:
            dict: Dictionary mit Metriken
        """
        if len(X.shape) != 3:
            raise ValueError(f"X muss 3-dimensional sein (batch_size, seq_length, features), aber hat Form {X.shape}")
            
        if len(y.shape) != 2:
            raise ValueError(f"y muss 2-dimensional sein (batch_size, targets), aber hat Form {y.shape}")
            
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            outputs, _ = self.model(X)  # Ignoriere den hidden state
            loss = self.criterion(outputs, y)
        
        return {'loss': loss.item()}
    
    def save_model(self, path):
        """
        Speichert das Modell.
        
        Args:
            path: Pfad zum Speichern des Modells
        """
        if self.rank == 0:  # Nur der Hauptprozess speichert
            torch.save(self.model.module.state_dict(), path)
            logger.info(f"Modell gespeichert unter {path}")
    
    def load_model(self, path):
        """
        Lädt ein gespeichertes Modell.
        
        Args:
            path: Pfad zum gespeicherten Modell
            
        Returns:
            Das geladene Modell
        """
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.module.load_state_dict(state_dict)
            logger.info(f"Modell geladen von {path}")
            return self.model.module
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            raise
    
    def cleanup(self):
        """Bereinigt die verteilte Umgebung."""
        dist.destroy_process_group()
        logger.info("Verteilte Umgebung bereinigt") 