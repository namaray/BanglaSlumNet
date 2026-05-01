# BanglaSlumNet: Implementation Documentation & Roadmap

**Project:** Physics-Guided Atmospheric Disentanglement With Cross-Modal Socioeconomic Fusion for Decadal Informal Settlement Mapping of Bangladesh  
**Status:** Stage 2 Cross-Attention Fusion Validated (Mission 3 Complete)

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
-[x] **Model Architecture:** Built Stage 1 (SAS-Net), Stage 2 (Cross-Attention Fusion), and Decoder (U-Net).
- [x] **Forward Pass:** End-to-end forward pass completed successfully without tensor dimension mismatches.

### Mission 2: Isolate & Train Stage 1 (The De-Hazer)
- [x] **Paired Data Prep:** Wrote GEE script to download aligned Hazy/Clear image pairs.
- [x] **Discriminator:** Built the `PatchGANDiscriminator` (70x70 receptive field).
- [x] **Loss Functions:** Implemented $L_{recon}$ (MSE), $L_{adv}$ (Adversarial), and $L_{scene}$ (Structure consistency L1 loss).
-[x] **Training Loop 1:** Wrote and successfully executed the GAN training loop. 

### Mission 3: Isolate & Train Stage 2 (Optical Ambiguity)
- [x] **Socioeconomic Data Prep:** Wrote GEE script to download Google Open Buildings (Density), VIIRS (Nighttime Lights), and WorldPop (Population) perfectly aligned to satellite tiles.
- [x] **Loss Functions:** Implemented $L_{IoU}$ (Soft IoU) and $L_{BCE}$ (Binary Cross-Entropy) with class weighting (3.0 for slums, 1.0 for background).
- [x] **Training Loop 2:** Wrote and successfully executed the multi-modal Cross-Attention training loop (`train_stage2.py`). Gradients passed cleanly through fusion and decoder.

---

## 📂 Current Codebase Files
1. `download_s2.py` - Connects to GEE and downloads single Sentinel-2 tiles.
2. `view_tile.py` - Uses `rasterio` and `matplotlib` to visualize raw satellite tensors.
3. `slum_dataloader.py` - PyTorch dataset class for handling raw GeoTIFFs.
4. `sasnet.py` - Holds the Structure Encoder, Appearance Encoder, and AdaIN modules.
5. `stage2_fusion.py` - Holds the Socioeconomic Encoders and Multi-Head Cross-Attention module.
6. `decoder.py` - Holds the U-Net upsampler and the master end-to-end test script.
7. `download_pairs.py` - Downloads perfectly aligned Hazy/Clear image pairs.
8. `stage1_gan.py` - Contains the PatchGAN Discriminator and custom `SASNetLoss` math.
9. `train_stage1.py` - The master training loop for Stage 1.
10. `download_socioeconomic.py` - Downloads aligned NTL, Pop, and GOB rasters via GEE.
11. `train_stage2.py` - The master training loop for Stage 2 (Cross-Attention).

---

## 🚀 Roadmap: Things Left To Do

### 🛠️ Urgent Infrastructure Fix (Pending Transfer)
- [ ] **CUDA Setup:** Acknowledge CPU bottleneck. Final training will be ported to an NVIDIA RTX 5060 Ti machine running PyTorch with CUDA 12.x support.

### Mission 4: Stage 3 (Temporal Smoothing)
*Goal: Ensure the 10-year maps don't flicker unnaturally.*
- [ ] **SAR Mask:** Generate the $M_{no-change}$ binary mask using Sentinel-1 VV/VH log-ratio change detection.
- [ ] **ConvLSTM:** Implement the 2D ConvLSTM equations.
- [ ] **Training Loop 3:** End-to-end fine-tuning with $L_{temp}$ loss.

### Mission 5: National Scale-up & Evaluation (The 1.8 TB Run)
*Goal: Produce the final outputs for the research paper.*
- [ ] Run inference over all 64 districts of Bangladesh for 2015-2025.
- [ ] Generate standard evaluation metrics (mIoU, F1, mAP) against the test set.
- [ ] Test Zero-Shot generalization on Khulna/Chittagong and international cities (Karachi/Mumbai).
- [ ] Export final GeoTIFFs, vector shapefiles, and uncertainty maps for Zenodo publication.