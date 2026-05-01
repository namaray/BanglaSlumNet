# BanglaSlumNet: Implementation Documentation & Roadmap

**Project:** Physics-Guided Atmospheric Disentanglement With Cross-Modal Socioeconomic Fusion for Decadal Informal Settlement Mapping of Bangladesh  
**Status:** Stage 3 ConvLSTM Temporal Engine Validated (Mission 4 Complete)

---

## 🎯 Project Overview
BanglaSlumNet is a 3-stage deep learning pipeline designed to map informal settlements across Bangladesh from 2015-2025. It uniquely addresses two core barriers in South Asian remote sensing:
1. **Atmospheric Haze:** Uses a modified SAS-Net (differentiable inverse rendering) to remove brick-kiln smog.
2. **Optical Ambiguity:** Uses Cross-Attention to fuse physical satellite structures with socioeconomic data (Google Open Buildings, Nighttime Lights, Poverty) to distinguish slums from dense formal housing.
3. **Temporal Inconsistency:** Uses a ConvLSTM regularized by Sentinel-1 SAR to prevent year-to-year prediction flickering.

---

## ✅ Milestones Achieved 

### Mission 1: Architecture Skeleton
- [x] **Cloud Setup:** Google Cloud Project registered and Earth Engine API authenticated.
- [x] **Data Ingestion:** Sentinel-2 GEE export script written and tested.
- [x] **PyTorch DataLoader:** Built `BanglaSlumDataset` to load `.tif` files and format them as `[1, 4, 512, 512]` tensors.
- [x] **Model Architecture:** Built Stage 1 (SAS-Net), Stage 2 (Cross-Attention Fusion), and Decoder (U-Net).
- [x] **Forward Pass:** End-to-end forward pass completed successfully without tensor dimension mismatches.

### Mission 2: Isolate & Train Stage 1 (The De-Hazer)
- [x] **Paired Data Prep:** Wrote GEE script to download aligned Hazy/Clear image pairs.
- [x] **Discriminator:** Built the `PatchGANDiscriminator` (70x70 receptive field).
- [x] **Loss Functions:** Implemented $L_{recon}$ (MSE), $L_{adv}$ (Adversarial), and $L_{scene}$ (Structure consistency L1 loss).
- [x] **Training Loop 1:** Wrote and successfully executed the GAN training loop. 

### Mission 3: Isolate & Train Stage 2 (Optical Ambiguity)
- [x] **Socioeconomic Data Prep:** Wrote GEE script to download Google Open Buildings (Density), VIIRS (Nighttime Lights), and WorldPop (Population) perfectly aligned to satellite tiles.
-[x] **Loss Functions:** Implemented $L_{IoU}$ (Soft IoU) and $L_{BCE}$ (Binary Cross-Entropy).
- [x] **Training Loop 2:** Wrote and successfully executed the multi-modal Cross-Attention training loop. 

### Mission 4: Stage 3 (Temporal Smoothing)
- [x] **Temporal Data Prep:** Wrote GEE script to download a 3-year optical sequence and the corresponding Sentinel-1 SAR radar change masks.
- [x] **ConvLSTM:** Implemented the 2D ConvLSTM cell and temporal sequence loop.
- [x] **Loss Functions:** Implemented the $L_{temp}$ loss, masking out penalties when SAR radar confirms physical construction/demolition.
- [x] **Training Loop 3:** Successfully trained the 5D time-sequence end-to-end. Gradients flowed through the time dimension seamlessly.

---

## 📂 Current Codebase Files
1. `download_s2.py` - Connects to GEE and downloads single Sentinel-2 tiles.
2. `view_tile.py` - Uses `rasterio` and `matplotlib` to visualize raw satellite tensors.
3. `slum_dataloader.py` - PyTorch dataset class for handling raw GeoTIFFs.
4. `sasnet.py` - Holds the Structure Encoder, Appearance Encoder, and AdaIN modules.
5. `stage1_gan.py` - Contains the PatchGAN Discriminator and custom `SASNetLoss` math.
6. `train_stage1.py` - The master training loop for Stage 1.
7. `download_pairs.py` - Downloads perfectly aligned Hazy/Clear image pairs.
8. `download_socioeconomic.py` - Downloads aligned NTL, Pop, and GOB rasters via GEE.
9. `stage2_fusion.py` - Holds the Socioeconomic Encoders and Multi-Head Cross-Attention module.
10. `train_stage2.py` - The master training loop for Stage 2 (Cross-Attention).
11. `download_temporal.py` - Downloads 3-year Sentinel-2 sequences and Sentinel-1 SAR masks.
12. `stage3_temporal.py` - Contains the 2D ConvLSTM and SAR-masked Temporal Loss equations.
13. `train_stage3.py` - The master 5D training loop for Stage 3.

---

## 🚀 Roadmap: Things Left To Do

### 🛠️ The Handoff Package (Pre-Training Phase)
- [ ] **Dependencies:** Generate `requirements.txt` for easy installation on the host machine.
- [ ] **CUDA Setup:** Host machine must install PyTorch with CUDA 12.x support to utilize the NVIDIA RTX 5060 Ti.

### Mission 5: Inference & Export (The Post-Training Phase)
*Goal: Turn AI probabilities into GIS-ready maps for the research paper.*
- [ ] **Inference Script:** Write a script to pass a raw tile through the trained pipeline.
- [ ] **Morphological Closing:** Implement the 5x5 smoothing filter (Blueprint Sec 3.6).
- [ ] **GIS Export:** Convert the binary mask back into a `.tif` file and a `.shp` vector polygon file using GDAL/Rasterio.
- [ ] **National Run:** Run the inference script over all 64 districts.