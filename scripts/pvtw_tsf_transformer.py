import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from models.timeseries_transformer import TimeSeriesTransformer

class PVTWTimeSeriesTransformer:
    def __init__(self, config, seed = 42):
        self.config = config 
        self.set_seed(seed)

        self.model = TimeSeriesTransformer(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            ff_dim=config['ff_dim'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
            threshold= config['threshold']
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 0.0001))
        self.epochs = config.get('epochs', 50)

    @staticmethod
    def set_seed(seed):
        """
        Set the seed for reproducibility.

        Parameters:
            seed (int): The random seed.
        """
        import torch
        import numpy as np
        import random

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seed set to {seed}")
    

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_x, batch_y)

            # Compute loss
            loss = self.criterion(outputs, batch_y)
            train_loss += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return train_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x, batch_y)

                # Compute loss
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def fit(self, train_loader, val_loader):
        """
        Train the model using the provided training data.

        Parameters:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validating data.
        """
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if epoch == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}/{self.epochs}, "
                      f"Training Loss: {train_loss:.6f}, "
                      f"Validation Loss: {val_loss:.6f}")

    def evaluate(self, test_loader):
        """
        Evaluate the model on the test dataset.

        Parameters:
            test_loader (DataLoader): DataLoader for testing data.

        Returns:
            dict: Dictionary containing MAE and RMSE metrics.
        """
        self.model.eval()
        all_predictions, all_actuals, all_inputs = [], [], []

        for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward pass
                output = self.model(batch_x, batch_y)
                predictions = output.squeeze(-1)  # [batch_size, output_seq_length]
                actuals = batch_y.squeeze(-1)    # [batch_size, output_seq_length]
                inputs = batch_x.squeeze(-1)     # [batch_size, input_seq_length]

                # Collect predictions, actuals, and inputs
                all_predictions.append(predictions.cpu())
                all_actuals.append(actuals.cpu())
                all_inputs.append(inputs.cpu())

        return torch.cat(all_predictions, dim=0), torch.cat(all_actuals, dim=0), torch.cat(all_inputs, dim=0)
    
    def save_model(self, path):
        """
        Save the trained model to the specified path.

        Parameters:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    
    def load_model(self, path):
        """
        Load a model from the specified path.

        Parameters:
            path (str): Path to the saved model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
