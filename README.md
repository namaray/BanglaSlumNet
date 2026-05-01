# BanglaSlumNet: Implementation Documentation & Roadmap

**Project:** Physics-Guided Atmospheric Disentanglement With Cross-Modal Socioeconomic Fusion for Decadal Informal Settlement Mapping of Bangladesh  
**Status:** Architecture Skeleton Validated (Mission 1 Complete)

---

## 🎯 Project Overview
BanglaSlumNet is a 3-stage deep learning pipeline designed to map informal settlements across Bangladesh from 2015-2025. It uniquely addresses two core barriers in South Asian remote sensing:
1. **Atmospheric Haze:** Uses a modified SAS-Net (differentiable inverse rendering) to remove brick-kiln smog.
2. **Optical Ambiguity:** Uses Cross-Attention to fuse physical satellite structures with socioeconomic data (Google Open Buildings, Nighttime Lights, Poverty) to distinguish slums from dense formal housing.
3. **Temporal Inconsistency:** Uses a ConvLSTM regularized by Sentinel-1 SAR to prevent year-to-year prediction flickering.

---

## ✅ Milestones Achieved (Mission 1)
*Goal: Prove mathematical soundness and data flow of the core architecture.*

- [x] **Cloud Setup:** Google Cloud Project registered (`banglaslumnet-research`) and Earth Engine API authenticated.
- [x] **Data Ingestion:** Script written to query GEE, filter clouds, composite, and download Sentinel-2 `.tif` tensors.
- [x] **PyTorch DataLoader:** Built `BanglaSlumDataset` to load `.tif` files, normalize physical radiance, and crop to exact `[1, 4, 512, 512]` tensors.
- [x] **Stage 1 (SAS-Net):** Built `StructureEncoder` (EfficientNet-B2 truncated at stride 8) and `AppearanceEncoder` with `AdaIN` style-injection.
- [x] **Stage 2 (Fusion):** Built `CrossAttentionFusion` module to project 5 socioeconomic rasters into Keys/Values and merge with Visual Structure (Query) via Multi-Head Attention.
- [x] **Decoder:** Built a 3-block bilinear U-Net `SegmentationDecoder` ending in a Sigmoid activation.
- [x] **Validation:** End-to-end forward pass completed successfully without tensor dimension mismatches. Output shape validated as `[Batch, 1, 512, 512]`.

### Current Codebase Files
1. `download_s2.py` - Connects to GEE and downloads Sentinel-2 tiles.
2. `view_tile.py` - Uses `rasterio` and `matplotlib` to visualize raw satellite tensors.
3. `slum_dataloader.py` - PyTorch dataset class for handling raw GeoTIFFs.
4. `sasnet.py` - Holds the Structure Encoder, Appearance Encoder, and AdaIN modules.
5. `stage2_fusion.py` - Holds the Socioeconomic Encoders and Multi-Head Cross-Attention module.
6. `decoder.py` - Holds the U-Net upsampler and the master end-to-end test script.

---

## 🚀 Roadmap: Things Left To Do

### Mission 2: Isolate & Train Stage 1 (The De-Hazer)
*Goal: Teach the model to see through Bangladesh's winter brick-kiln smog.*
- [ ] **Data Prep:** Write a GEE script to download ~100-500 *paired* images of Dhaka (Hazy Winter image vs. Clear Spring image of the exact same location).
- [ ] **Discriminator:** Build the `PatchGAN` Discriminator network.
- [ ] **Loss Functions:** Implement $L_{recon}$ (MSE), $L_{adv}$ (Adversarial), and $L_{scene}$ (Structure consistency) from the blueprint.
- [ ] **Training Loop 1:** Train Stage 1 in isolation until it can successfully take a hazy image and output a clean one. *Freeze weights once complete.*

### Mission 3: Isolate & Train Stage 2 (Optical Ambiguity)
*Goal: Teach the model to distinguish slums from formal housing using socioeconomic clues.*
- [ ] **Socioeconomic Data Prep:** Download and align Google Open Buildings (Density), VIIRS (Nighttime Lights), and Poverty Maps for Dhaka.
- [ ] **Label Prep:** Ingest the GRAM dataset labels for Dhaka as the ground-truth targets (`y`).
- [ ] **Loss Functions:** Implement $L_{IoU}$ (Soft IoU) and $L_{BCE}$ (Binary Cross-Entropy) with class weighting (3.0 for slums, 1.0 for background).
- [ ] **Training Loop 2:** Train the Cross-Attention and Decoder modules on the clear Sentinel-2 images + Socioeconomic stack.

### Mission 4: Stage 3 (Temporal Smoothing)
*Goal: Ensure the 10-year maps don't flicker unnaturally.*
- [ ] **Temporal Data Prep:** Extract a 10-year sequence of feature maps for specific test locations.
- [ ] **SAR Mask:** Generate the $M_{no-change}$ binary mask using Sentinel-1 VV/VH log-ratio change detection.
- [ ] **ConvLSTM:** Implement the 2D ConvLSTM equations.
- [ ] **Training Loop 3:** End-to-end fine-tuning with $L_{temp}$ loss.

### Mission 5: National Scale-up & Evaluation (The 1.8 TB Run)
*Goal: Produce the final outputs for the research paper.*
- [ ] Run inference over all 64 districts of Bangladesh for 2015-2025.
- [ ] Generate standard evaluation metrics (mIoU, F1, mAP) against the test set.
- [ ] Test Zero-Shot generalization on Khulna/Chittagong and international cities (Karachi/Mumbai).
- [ ] Export final GeoTIFFs, vector shapefiles, and uncertainty maps for Zenodo publication.