"""
TFT Model Trainer
PyTorch Lightning Trainer setup for Temporal Fusion Transformer
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from pytorch_lightning.loggers import TensorBoardLogger
import os
from typing import Dict, Any, Optional

class TFTTrainer:
    """Temporal Fusion Transformer Trainer"""
    
    def __init__(self, 
                 model_params: Dict[str, Any],
                 training_params: Dict[str, Any],
                 output_dir: str = "./models"):
        self.model_params = model_params
        self.training_params = training_params
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def create_model(self, training_dataset) -> TemporalFusionTransformer:
        """
        Create TFT model with specified parameters
        
        Args:
            training_dataset: TimeSeriesDataSet for training
            
        Returns:
            Configured TFT model
        """
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **self.model_params
        )
        
        return model
    
    def setup_trainer(self, 
                     max_epochs: int = 100,
                     patience: int = 10,
                     learning_rate: float = 0.03) -> pl.Trainer:
        """
        Setup PyTorch Lightning trainer with callbacks
        
        Args:
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            learning_rate: Learning rate for optimizer
            
        Returns:
            Configured PyTorch Lightning trainer
        """
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=patience,
            verbose=False,
            mode="min"
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir,
            filename="tft-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=self.output_dir,
            name="tft_logs"
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            accelerator="auto",
            devices="auto",
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        return trainer
    
    def train_model(self, 
                   model: TemporalFusionTransformer,
                   train_dataloader,
                   val_dataloader,
                   trainer: pl.Trainer) -> Dict[str, Any]:
        """
        Train the TFT model
        
        Args:
            model: TFT model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            trainer: PyTorch Lightning trainer
            
        Returns:
            Training results dictionary
        """
        # Configure model for training
        model.configure_optimizers()
        
        # Train the model
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        # Get best model path
        best_model_path = trainer.checkpoint_callback.best_model_path
        
        # Load best model
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "tft_final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        
        return {
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "best_val_loss": trainer.checkpoint_callback.best_model_score.item(),
            "total_epochs": trainer.current_epoch + 1
        }
    
    def evaluate_model(self, 
                      model: TemporalFusionTransformer,
                      test_dataloader) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            model: Trained TFT model
            test_dataloader: Test data loader
            
        Returns:
            Evaluation metrics dictionary
        """
        model.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                # Get predictions
                pred = model.predict(batch, mode="prediction")
                predictions.extend(pred.cpu().numpy().flatten())
                
                # Get actual values
                actual = batch["decoder_target"]
                actuals.extend(actual.cpu().numpy().flatten())
        
        # Calculate metrics
        predictions = torch.tensor(predictions)
        actuals = torch.tensor(actuals)
        
        # MAE
        mae = torch.mean(torch.abs(predictions - actuals)).item()
        
        # RMSE
        rmse = torch.sqrt(torch.mean((predictions - actuals) ** 2)).item()
        
        # SMAPE
        smape = SMAPE()(predictions, actuals).item()
        
        # Quantile Loss (for different quantiles)
        quantile_loss = QuantileLoss()
        q_loss = quantile_loss(predictions, actuals).item()
        
        return {
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "quantile_loss": q_loss
        }
    
    def predict_future(self, 
                      model: TemporalFusionTransformer,
                      prediction_dataset,
                      n_predictions: int = 30) -> torch.Tensor:
        """
        Make future predictions
        
        Args:
            model: Trained TFT model
            prediction_dataset: Dataset for prediction
            n_predictions: Number of future predictions
            
        Returns:
            Future predictions tensor
        """
        model.eval()
        
        with torch.no_grad():
            predictions = model.predict(
                prediction_dataset, 
                mode="prediction",
                return_x=True
            )
        
        return predictions

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "hidden_size": 16,
    "lstm_layers": 1,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "output_size": 7,  # 7 quantiles
    "loss": QuantileLoss(),
    "reduce_on_plateau_patience": 4,
}

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "max_epochs": 100,
    "patience": 10,
    "learning_rate": 0.03,
    "batch_size": 64,
}
