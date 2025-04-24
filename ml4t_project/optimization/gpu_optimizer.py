"""
GPU-Optimierer für das ML4T-Projekt.
"""
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime, timedelta

class GPUOptimizer:
    def __init__(self,
                 model: torch.nn.Module,
                 use_mixed_precision: bool = True,
                 memory_fraction: float = 0.8,
                 batch_size: Optional[int] = None):
        """
        Initialisiert den GPU-Optimierer.
        
        Args:
            model: PyTorch-Modell
            use_mixed_precision: Mixed Precision Training aktivieren
            memory_fraction: Maximaler GPU-Speicheranteil
            batch_size: Optionale Batch-Größe
        """
        self.model = model
        self.use_mixed_precision = use_mixed_precision
        self.memory_fraction = memory_fraction
        self.batch_size = batch_size
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
        
        # Prüfe GPU-Verfügbarkeit
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"GPU gefunden: {torch.cuda.get_device_name()}")
            self._setup_gpu()
        else:
            self.logger.warning("Keine GPU verfügbar")

    def _setup_gpu(self):
        """Richtet GPU-Optimierungen ein."""
        # Setze Speicherlimit
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Aktiviere Autotuner
            torch.backends.cudnn.benchmark = True
            
            # Mixed Precision
            if self.use_mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler()
            
            self.logger.info(
                f"GPU Setup: {self.memory_fraction*100:.0f}% Speicher "
                f"({self.memory_fraction*total_memory/1e9:.1f}GB)"
            )

    def optimize_model(self):
        """Optimiert das Modell für GPU-Ausführung."""
        if self.device.type != 'cuda':
            return
            
        try:
            # Verschiebe Modell auf GPU
            self.model = self.model.to(self.device)
            
            # Optimiere Modellarchitektur
            self._optimize_architecture()
            
            # Optimiere Speichernutzung
            self._optimize_memory()
            
            self.logger.info("Modell für GPU optimiert")
        except Exception as e:
            self.logger.error(f"Fehler bei Modelloptimierung: {e}")

    def _optimize_architecture(self):
        """Optimiert die Modellarchitektur."""
        # Aktiviere Fusion von Operationen
        torch.backends.cudnn.benchmark = True
        
        # Verwende optimierte Kernel wenn möglich
        if hasattr(self.model, 'fuse_model'):
            self.model.fuse_model()
            
        # Optimiere Datentransfers
        torch.cuda.set_device(self.device)

    def _optimize_memory(self):
        """Optimiert die Speichernutzung."""
        # Bereinige nicht benötigten Speicher
        torch.cuda.empty_cache()
        
        # Optimiere Gradienten-Speicherung
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = None

    def train_step(self,
                  inputs: torch.Tensor,
                  targets: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> Tuple[float, Dict[str, float]]:
        """
        Führt einen optimierten Trainingsschritt aus.
        
        Args:
            inputs: Eingabedaten
            targets: Zielwerte
            optimizer: Optimizer-Instanz
            
        Returns:
            Tuple aus (Loss, Metriken)
        """
        if self.device.type != 'cuda':
            return self._train_step_cpu(inputs, targets, optimizer)
            
        try:
            # Verschiebe Daten auf GPU
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed Precision Training
            if self.use_mixed_precision:
                return self._train_step_mixed_precision(
                    inputs, targets, optimizer
                )
            else:
                return self._train_step_full_precision(
                    inputs, targets, optimizer
                )
                
        except Exception as e:
            self.logger.error(f"Fehler im Trainingsschritt: {e}")
            raise

    def _train_step_cpu(self,
                       inputs: torch.Tensor,
                       targets: torch.Tensor,
                       optimizer: torch.optim.Optimizer) -> Tuple[float, Dict[str, float]]:
        """
        Führt einen CPU-Trainingsschritt aus.
        
        Args:
            inputs: Eingabedaten
            targets: Zielwerte
            optimizer: Optimizer-Instanz
            
        Returns:
            Tuple aus (Loss, Metriken)
        """
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item(), {'loss': loss.item()}

    def _train_step_mixed_precision(self,
                                  inputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  optimizer: torch.optim.Optimizer) -> Tuple[float, Dict[str, float]]:
        """
        Führt einen Mixed Precision Trainingsschritt aus.
        
        Args:
            inputs: Eingabedaten
            targets: Zielwerte
            optimizer: Optimizer-Instanz
            
        Returns:
            Tuple aus (Loss, Metriken)
        """
        optimizer.zero_grad()
        
        # Forward Pass mit Mixed Precision
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # Backward Pass mit Gradient Scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item(), {'loss': loss.item()}

    def _train_step_full_precision(self,
                                 inputs: torch.Tensor,
                                 targets: torch.Tensor,
                                 optimizer: torch.optim.Optimizer) -> Tuple[float, Dict[str, float]]:
        """
        Führt einen Full Precision Trainingsschritt aus.
        
        Args:
            inputs: Eingabedaten
            targets: Zielwerte
            optimizer: Optimizer-Instanz
            
        Returns:
            Tuple aus (Loss, Metriken)
        """
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item(), {'loss': loss.item()}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Gibt Performance-Metriken zurück.
        
        Returns:
            Dictionary mit Metriken
        """
        if self.device.type != 'cuda':
            return {}
            
        metrics = {
            'gpu_utilization': torch.cuda.utilization(),
            'memory_allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'memory_cached': torch.cuda.memory_reserved() / 1e9,  # GB
            'max_memory_allocated': torch.cuda.max_memory_allocated() / 1e9  # GB
        }
        
        if self.use_mixed_precision:
            metrics['mixed_precision'] = True
            
        return metrics

    def get_recommendations(self) -> List[str]:
        """
        Generiert Optimierungsempfehlungen.
        
        Returns:
            Liste von Empfehlungen
        """
        if self.device.type != 'cuda':
            return ["GPU nicht verfügbar - keine Optimierungen möglich"]
            
        recommendations = []
        metrics = self.get_performance_metrics()
        
        # Speichernutzung
        memory_usage = metrics['memory_allocated'] / (
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
        if memory_usage > 0.9:
            recommendations.append(
                f"Hohe Speicherauslastung ({memory_usage*100:.1f}%) - "
                "Batch-Größe reduzieren oder Gradient Checkpointing verwenden"
            )
            
        # GPU-Auslastung
        if metrics['gpu_utilization'] < 50:
            recommendations.append(
                f"Niedrige GPU-Auslastung ({metrics['gpu_utilization']}%) - "
                "Batch-Größe erhöhen oder Daten-Pipeline optimieren"
            )
            
        # Mixed Precision
        if not self.use_mixed_precision:
            recommendations.append(
                "Mixed Precision Training aktivieren für bessere Performance"
            )
            
        # Speicher-Caching
        if metrics['memory_cached'] > metrics['memory_allocated'] * 2:
            recommendations.append(
                "Hoher Cache-Speicher - torch.cuda.empty_cache() aufrufen"
            )
            
        return recommendations

    def cleanup(self):
        """Räumt GPU-Ressourcen auf."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if hasattr(self, 'model'):
                self.model.cpu()
            self.logger.info("GPU-Ressourcen aufgeräumt") 