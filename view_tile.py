import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 1. Open the satellite image
image_path = 'dhaka_sentinel2_tile.tif'
with rasterio.open(image_path) as src:
    # Read the 4 bands (Red, Green, Blue, NIR)
    # rasterio reads as (Channels, Height, Width)
    img_tensor = src.read()

print(f"✅ Successfully loaded image!")
print(f"Shape of the data: {img_tensor.shape} (Channels, Height, Width)")

# 2. Extract the RGB bands to show a normal photo
# Bands are 1-indexed in rasterio. 
# In our download: 1=Red, 2=Green, 3=Blue, 4=NIR
red = img_tensor[0]
green = img_tensor[1]
blue = img_tensor[2]

# Stack them into a normal image shape (Height, Width, Channels)
rgb_image = np.dstack((red, green, blue))

# 3. Satellite images are dark by default (raw physics data). 
# We normalize the brightness so our human eyes can see it.
rgb_image = rgb_image / 3000.0  # 3000 is a standard Sentinel-2 brightness cap
rgb_image = np.clip(rgb_image, 0, 1)

# 4. Show the image!
plt.figure(figsize=(8, 8))
plt.imshow(rgb_image)
plt.title("Dhaka Sentinel-2 Tile (512x512)")
plt.axis('off')
plt.show()