import torch
import torch.nn as nn
import pytorch_lightning as pl

class PinballLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: (batch, horizon, len(quantiles))
        target: (batch, horizon)
        """
        loss = 0.0
        target = target.unsqueeze(-1) # (batch, horizon, 1)
        for i, q in enumerate(self.quantiles):
            diff = target[..., 0] - preds[..., i]
            loss += torch.where(diff >= 0, q * diff, (1 - q) * (-diff)).mean()
        return loss / len(self.quantiles)

class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self, num_features: int = 10, d_model: int = 64, nhead: int = 4, num_layers: int = 2, 
                 horizon: int = 7, quantiles=[0.1, 0.5, 0.9], lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles
        self.lr = lr
        
        # Input embedding to d_model
        self.input_linear = nn.Linear(num_features, d_model)
        
        # Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: maps from d_model at the last step to (horizon * num_quantiles)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon * len(quantiles))
        )
        
        self.loss_fn = PinballLoss(quantiles=quantiles)

    def forward(self, x):
        """
        x: (batch, context_length, num_features)
        """
        # (batch, context_length, d_model)
        embedded = self.input_linear(x)
        
        # (batch, context_length, d_model)
        encoded = self.transformer_encoder(embedded)
        
        # We take the representation of the last time step to predict the horizon
        # (batch, d_model)
        last_step_encoded = encoded[:, -1, :]
        
        # (batch, horizon * len(quantiles))
        out = self.output_head(last_step_encoded)
        
        # Reshape to (batch, horizon, len(quantiles))
        out = out.view(x.size(0), self.hparams.horizon, len(self.quantiles))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
