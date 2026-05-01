import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os

class BanglaSlumDataset(Dataset):
    def __init__(self, image_paths):
        """
        image_paths: A list of file paths to our .tif images
        """
        self.image_paths = image_paths

    def __len__(self):
        # Tells PyTorch how many images we have in total
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Tells PyTorch how to open ONE specific image
        img_path = self.image_paths[idx]
        
        with rasterio.open(img_path) as src:
            # Read all 4 bands
            img_array = src.read() 
        
        # 1. Convert to float32 (Neural Networks hate integers)
        img_array = img_array.astype(np.float32)
        
        # 2. Normalize the data to be between 0 and 1
        # Sentinel-2 raw values usually max out around 3000 for normal land
        img_array = img_array / 3000.0
        img_array = np.clip(img_array, 0.0, 1.0)
        
        # 3. Convert the numpy array into a PyTorch Tensor
        tensor_image = torch.from_numpy(img_array)
        
        # --- NEW LINE: Force exact 512x512 crop ---
        tensor_image = tensor_image[:, :512, :512]
        
        return tensor_image
# --- Let's test it! ---
if __name__ == "__main__":
    # We only have 1 image right now, so we make a list of just one file
    my_files =['dhaka_sentinel2_tile.tif']
    
    # 1. Create the Dataset
    my_dataset = BanglaSlumDataset(image_paths=my_files)
    
    # 2. Create the DataLoader (We tell it to grab 1 image per batch)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=False)
    
    print("✅ DataLoader created successfully!")
    
    # 3. Fetch the first batch of data
    for batch in my_dataloader:
        print(f"Data type: {type(batch)}")
        print(f"Batch Shape: {batch.shape} -> [Batch_Size, Channels, Height, Width]")
        print(f"Max pixel value: {batch.max():.4f}")
        print(f"Min pixel value: {batch.min():.4f}")
        break  # We only want to test the first batch