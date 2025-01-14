from torch.utils.data import Dataset
import torch

class SequentialDataset(Dataset):
    def __init__(self, data, timestamps, input_window=24, output_window=12):
        """
        A custom dataset class for handling sequential data. It creates input-output pairs based on sliding windows.
        
        Arguments:
        - data: The dataset (list or tensor) containing the sequences.
        - seq_len: Length of the input sequence.
        - label_len: Length of the output (label) sequence. Default is 5.
        """
        self.data = data
        self.timestamps = timestamps
        self.seq_len = input_window
        self.label_len = output_window
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.label_len
    
    def __getitem__(self, index):
        assert index + self.seq_len + self.label_len <= len(self.data), \
            f"Index {index} is out of bounds for data with length {len(self.data)}."

        # Create sequences from the data
        seq_x = self.data[index:index + self.seq_len]
        seq_y = self.data[index + self.seq_len:index + self.seq_len + self.label_len]

        seq_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1) # Ensure it's a tensor
        seq_y = torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1)  # Ensure it's a tensor
        
        return seq_x, seq_y
    
    def get_datetime_sequences(self, index):
        """
        Get the datetime sequences for the given index.
        
        Arguments:
        - index: The starting index for the sequence.
        
        Returns:
        - seq_x_timestamps: The timestamps for the input sequence.
        - seq_y_timestamps: The timestamps for the output sequence.
        """
        assert index + self.seq_len + self.label_len <= len(self.timestamps), \
            f"Index {index} is out of bounds for timestamps with length {len(self.timestamps)}."

        seq_x_timestamps = self.timestamps[index:index + self.seq_len]
        seq_y_timestamps = self.timestamps[index + self.seq_len:index + self.seq_len + self.label_len]
        
        return seq_x_timestamps, seq_y_timestamps


