# BanglaSlumNet Presentation Script

Speaker assignment:

| Slides | Speaker |
|---|---|
| 1-4 | Nafiz |
| 5-8 | Zayan |
| 9-12 | Mizi |
| 13-17 | Fardeen |
| 18-21 | Namare |

---

# Slides 1-4: Nafiz

## Slide 1 - BanglaSlumNet

**5-line presentation script**

1. Good morning everyone, today we are presenting BanglaSlumNet, our project on slum detection in Dhaka using language-grounded socioeconomic fusion.
2. The main question we ask is whether we can separate informal density from formal density in a complex megacity.
3. Dhaka is a strong case study because dense slums and dense formal neighborhoods can look very similar from satellite imagery.
4. Our pipeline combines LocateAnything, a vision-language grounding model, with socioeconomic data such as nightlight, population, roads, and built-up layers.
5. The goal is not only to detect dense roofs, but to understand whether that density represents informality.

**Technical terms explained**

- **Language-grounded**: The model is guided by text descriptions, such as "informal settlement" or "formal dense housing", instead of only learning numeric class labels.
- **Socioeconomic fusion**: Combining satellite imagery with non-visual indicators such as population, roads, nightlight, and poverty proxies.
- **Dense megacity**: A very large city with high population and building density, such as Dhaka.
- **Informal density vs. formal density**: Slums and formal neighborhoods can both be dense; the challenge is to distinguish their social and infrastructural meaning.

---

## Slide 2 - Why Slum Detection Matters

**5-line presentation script**

1. Slum detection matters because informal settlements are directly connected to urban planning, public health, disaster response, and infrastructure allocation.
2. In fast-growing cities, official maps are often outdated, incomplete, or too slow to update.
3. Ground surveys can be accurate, but they are expensive and difficult to repeat at city scale.
4. Remote sensing gives us regular city-wide coverage, but it cannot fully capture socioeconomic reality from roofs alone.
5. So our project treats slum mapping as both a spatial problem and a social-context problem.

**Technical terms explained**

- **Remote sensing**: Collecting information about Earth from satellites, aircraft, or drones without physically visiting the location.
- **Temporal consistency**: The ability to observe the same area repeatedly over time, such as every year or season.
- **Spatial reach**: The ability to cover a large geographic area.
- **Socioeconomic reality**: The social and economic condition of a place, including poverty, services, roads, and infrastructure access.

---

## Slide 3 - The Dhaka Challenge: Density Is Not Informality

**5-line presentation script**

1. Dhaka is difficult because it is dense almost everywhere, not only in informal settlements.
2. Places like Korail and Kamrangirchar can visually resemble dense formal areas such as Old Dhaka or Dhanmondi in satellite images.
3. Optical-only models often make the shortcut that dense built-up texture means slum.
4. This creates many false positives in formal neighborhoods that are dense but not informal settlements.
5. Our key argument is that visual density alone is an unreliable proxy for socioeconomic condition.

**Technical terms explained**

- **Optical-only model**: A model that uses only visual satellite bands, such as RGB or multispectral imagery, without outside context.
- **False positive**: A wrong prediction where the model says "slum" for an area that is not actually slum.
- **Proxy**: An indirect indicator. For example, density can be a proxy for urban development, but not always for informality.
- **Morphology**: The visible shape and arrangement of buildings, roads, and urban blocks.

---

## Slide 4 - Research Questions

**5-line presentation script**

1. Our first research question asks how we can distinguish informal settlements from dense formal neighborhoods when both look similar in satellite imagery.
2. Our second question asks whether language grounding can separate the concept of "informal settlement" from general dense built-up texture.
3. Our third question asks whether socioeconomic signals can reduce false positives in formal dense regions.
4. These questions guide the full design of BanglaSlumNet: visual features, language prompts, and socioeconomic fusion.
5. In short, we are testing whether extra context can solve a failure that raw imagery alone cannot solve.

**Technical terms explained**

- **Language grounding**: Connecting words or prompts to visual regions in an image.
- **Dense built-up urban fabric**: A city area with many buildings packed close together.
- **Nighttime lights**: Satellite-measured light at night, often used as a proxy for electricity access and economic activity.
- **Road access**: A measure of how connected an area is to formal transport or street networks.

---

# Slides 5-8: Zayan

## Slide 5 - Earlier Remote-Sensing Approaches

**5-line presentation script**

1. Earlier slum mapping methods started with handcrafted features, where experts designed texture, spectral, and shape indicators manually.
2. Classical machine learning improved automation, but still depended heavily on manually designed filters.
3. CNN and UNet models learned spatial patterns more automatically, but they require good pixel-level labels.
4. Foundation models brought stronger generalization, but they can still learn biased shortcuts such as "dense equals slum".
5. This progression shows why we need a method that is both automated and context-aware.

**Technical terms explained**

- **Handcrafted features**: Manually designed measurements, such as texture roughness or roof density.
- **Classical ML**: Traditional machine learning methods such as random forests, SVMs, or logistic regression.
- **CNN**: Convolutional Neural Network, a deep learning model commonly used for images.
- **UNet**: A popular neural network architecture for pixel-level segmentation.
- **Foundation model**: A large pre-trained model that can be adapted to many tasks.

---

## Slide 6 - Foundation Models and Optical-Only Failure

**5-line presentation script**

1. Foundation models are powerful because they learn from large datasets and transfer to new areas.
2. However, when they depend only on optical imagery, they may still rely on visual shortcuts.
3. In many datasets, crowded roofs are associated with poverty, so the model may learn density instead of informality.
4. In Dhaka, that shortcut fails because dense planned districts and dense informal settlements can look similar.
5. This is why our project explicitly targets shortcut bias in optical-only slum detection.

**Technical terms explained**

- **Shortcut bias**: When a model learns an easy pattern that works in training but fails in real cases.
- **Optical bounds**: The limits of what can be inferred from visible or multispectral image data alone.
- **Inference chain**: The step-by-step reasoning path from input image to model prediction.
- **Morphological ambiguity**: When different urban areas have similar visible shapes or structures.

---

## Slide 7 - Vision-Language Models as a New Signal

**5-line presentation script**

1. Vision-language models add a new signal because they connect images with text descriptions.
2. Instead of giving the model only pixels, we can prompt it with concepts like "dense informal settlement" or "planned formal housing".
3. In our project, LocateAnything is the vision-language model used to produce grounded heatmap features from those prompts.
4. LocateAnything takes an image and a text prompt, identifies regions matching the prompt, and outputs grounding information that we cache as features.
5. This allows BanglaSlumNet to test whether semantic descriptions help separate slums from dense formal neighborhoods.

**Technical terms explained**

- **Vision-Language Model (VLM)**: A model trained to connect visual information with natural language text.
- **LocateAnything**: NVIDIA's vision-language grounding model. In our pipeline, it reads a satellite tile plus a text prompt and produces prompt-specific visual grounding features.
- **Zero-shot**: Using a model on a new task without training it specifically for that task.
- **Prompt**: The text instruction or description given to a language or vision-language model.
- **Grounded heatmap**: A spatial map showing where the model thinks the prompted concept appears in the image.

---

## Slide 8 - BanglaSlumNet System Overview

**5-line presentation script**

1. This slide summarizes the full BanglaSlumNet architecture.
2. The Sentinel-2 tile first goes through preprocessing and SAS-Net, which helps normalize scene appearance.
3. LocateAnything produces language-grounded visual features from prompts like slum and formal dense housing.
4. Socioeconomic layers are added through a fusion module so the model can use non-visual context.
5. Finally, the decoder outputs a slum probability mask for the tile.

**Technical terms explained**

- **Architecture**: The structure of the whole model or system.
- **Sentinel-2**: A European satellite mission that provides multispectral Earth imagery at around 10 m resolution for key bands.
- **SAS-Net**: Scene-Appearance Separation Network; in our project, it is used to normalize scene appearance before downstream prediction.
- **Decoder**: The final neural-network component that turns learned features into a prediction mask.
- **Probability mask**: A pixel map where each value indicates how likely that pixel is to belong to the target class.

---

# Slides 9-12: Mizi

## Slide 9 - Language Grounding in BanglaSlumNet

**5-line presentation script**

1. Language grounding helps the model focus on the meaning of urban patterns, not only their visual texture.
2. We use visual baseline prompts to extract general urban features without directly saying "slum".
3. We also use active slum and formal prompts to tell the model what distinction matters.
4. The outputs of the VLM are cached to Drive, so we do not run the large model during every training pass.
5. This makes the language ablation possible while keeping the experiment computationally manageable.

**Technical terms explained**

- **Visual baseline prompt**: A neutral prompt used to extract generic visual features without the slum concept.
- **Active prompt**: A prompt that directly describes the concept we want the model to identify.
- **Ablation**: An experiment where one component is removed or changed to measure its effect.
- **Caching**: Saving computed outputs so they can be reused without recomputing.
- **Drive**: Google Drive storage used here for persistent data and model artifacts across Colab sessions.

---

## Slide 10 - Socioeconomic Fusion

**5-line presentation script**

1. Socioeconomic fusion adds information that satellite RGB texture cannot directly show.
2. VIIRS nightlight can indicate electricity access and economic activity.
3. WorldPop and GHSL provide population and built-up context for the settlement.
4. Road maps or access proxies help represent how connected or planned an area is.
5. These layers are fused with visual features so the model can interpret dense urban texture using social and infrastructure context.

**Technical terms explained**

- **VIIRS**: A satellite sensor that measures nighttime light, often used as a proxy for electricity and economic activity.
- **WorldPop**: A gridded population dataset estimating population distribution.
- **GHSL**: Global Human Settlement Layer, which provides built-up and population-related global settlement data.
- **Road/access proxy**: An indirect dataset used to represent road connectivity when direct road data is unavailable.
- **Fusion**: Combining multiple data sources inside a model.

---

## Slide 11 - Model Variants We Compare

**5-line presentation script**

1. We compare multiple model variants to understand which signal actually helps.
2. The baseline CNN uses only Sentinel-2 imagery, so it represents the optical-only baseline.
3. The `vlm_visual` model uses VLM visual features but without the slum/formal language distinction.
4. The `vlm_lang` model adds active slum and formal prompts to isolate the effect of language.
5. The full system combines language-grounded VLM features with socioeconomic fusion.

**Technical terms explained**

- **baseline_cnn**: The simpler optical-only model used as a performance floor.
- **vlm_visual**: A model using cached VLM visual features from a neutral prompt.
- **vlm_lang**: A model using language-conditioned VLM features from slum/formal prompts.
- **full_system / full**: The proposed model using VLM language features plus socioeconomic fusion.
- **Performance floor**: A baseline result that stronger models should beat.

---

## Slide 12 - Study Area and Dataset

**5-line presentation script**

1. Our study area includes 12 specific regions across Dhaka.
2. Eight regions are informal target zones such as Korail, Bhashantek, and Kamrangirchar.
3. Four regions are formal dense controls such as Old Dhaka, Gulshan-Baridhara, Dhanmondi, and Uttara.
4. The dataset contains 720 Sentinel-2 tiles at 128 by 128 pixels and 10 m resolution.
5. The split is stratified by region so each split contains representation from the different area types.

**Technical terms explained**

- **Formal control region**: A dense area that should not be classified as slum; used to test false positives.
- **Tile**: A small cropped image patch used as one model input.
- **10 m resolution**: One pixel represents roughly 10 meters on the ground.
- **Stratified split**: A train/validation/test split that preserves representation from important groups.
- **Target zone**: An area known or assumed to contain the class of interest, here informal settlements.

---

# Slides 13-17: Fardeen

## Slide 13 - Weak Label Construction

**5-line presentation script**

1. For labels, we first tried using nighttime light thresholds to separate slum and formal pixels.
2. That approach failed because it collapsed label variation and produced zero slum pixels.
3. We then switched to region-type weak supervision using known region boundaries and built-up masks.
4. Built pixels inside informal regions are labeled slum, while built pixels inside formal control regions are labeled formal.
5. This gives us usable weak labels, although they are not the same as manual pixel-level ground truth.

**Technical terms explained**

- **Weak label**: A label created from indirect rules or assumptions, not manual annotation.
- **Region-type supervision**: Labeling pixels based on the known type of the region they belong to.
- **Built-up mask**: A map showing which pixels contain buildings or urban structures.
- **Dynamic World**: A Google land-cover dataset used to identify built-up areas.
- **Manual ground truth**: Labels created by human experts, usually more reliable but expensive.

---

## Slide 14 - Phase 1: Data Export and Region Benchmark

**5-line presentation script**

1. Phase 1 was about building the actual Dhaka benchmark and exporting the data.
2. We selected 12 regions so the model would see both informal settlements and difficult dense formal controls.
3. From Google Earth Engine, we exported Sentinel-2 dry-season imagery, weak-label rasters, and socioeconomic layers.
4. Then we tiled all rasters into aligned 128-pixel samples so every layer shares the same grid.
5. This phase gave us the 720-tile dataset used in the later experiments.

**Technical terms explained**

- **Google Earth Engine (GEE)**: A cloud platform for processing satellite and geospatial datasets.
- **Raster**: A grid-based geospatial image where each pixel stores a value.
- **Aligned grid**: All layers have the same pixel positions, size, and georeference.
- **Dry-season composite**: A cloud-reduced image created from dry-season satellite observations.
- **Benchmark**: A defined dataset and evaluation setup used to compare models.

---

## Slide 15 - Phase 2: Weak Labels and Label Verification

**5-line presentation script**

1. Phase 2 focused on making sure our labels were usable before spending GPU compute.
2. After the nighttime threshold failed, we used region-type weak labels combined with built-up masks.
3. The verification gate confirmed 720 tiles, over 11 million high-confidence pixels, and both slum and formal classes.
4. This fixed the earlier zero-slum bug that caused all-zero metrics.
5. However, these labels are still weak, because they label all built pixels in a region the same way.

**Technical terms explained**

- **High-confidence pixels (HC pixels)**: Pixels considered reliable enough for training or evaluation under the weak-label rule.
- **Class 1 / Class 2**: Numeric label encoding; here Class 1 is slum and Class 2 is formal.
- **Verification gate**: A pre-training check that stops the experiment if labels are missing or degenerate.
- **Degenerate labels**: Broken labels that contain only one class or no useful target pixels.
- **Pixel-level label**: A class assigned to each individual pixel in an image.

---

## Slide 16 - Phase 3: Feature and SAS-Net Caching

**5-line presentation script**

1. Phase 3 was the compute-heavy feature caching phase.
2. LocateAnything and MoonViT features were extracted once per tile and prompt, then stored in `data/features_cache`.
3. SAS-Net was also trained once to produce clean normalized tiles.
4. After this, Phase 4 can train small heads over cached tensors instead of calling the full VLM every epoch.
5. This design makes the experiment resumable and protects our Colab compute units.

**Technical terms explained**

- **MoonViT**: The vision encoder inside LocateAnything that extracts visual representations from images.
- **Grounding-map feature**: A map derived from where the VLM thinks a prompted object or concept appears.
- **Tensor**: A multidimensional numeric array used by neural networks.
- **Clean tile**: A normalized version of an image tile produced by SAS-Net.
- **Colab compute units (CU)**: Google Colab's paid compute usage budget.

---

## Slide 17 - Phase 4: Experiments We Intended to Run

**5-line presentation script**

1. Phase 4 was designed to run the experimental matrix and compare model variants.
2. The full plan included SAS-Net ablation, fusion ablation, and leave-one-region-out generalization.
3. For the current safe run, we focused on Experiment 2, the central fusion ablation.
4. We track not only HC-IoU, but also balanced accuracy, predicted-positive rate, formal-control FPR, precision, recall, and F1.
5. These extra diagnostics are important because HC-IoU alone initially rewarded trivial predictions.

**Technical terms explained**

- **Experiment matrix**: A list of model configurations that are trained and compared.
- **HC-IoU**: Intersection-over-Union measured on high-confidence pixels.
- **Balanced accuracy**: Average of performance on positive and negative classes; useful when classes are imbalanced.
- **Predicted-positive rate**: The fraction of pixels predicted as slum; used to detect all-slum or no-slum collapse.
- **FPR**: False Positive Rate; for us, how often formal control pixels are wrongly predicted as slum.
- **LORO**: Leave-One-Region-Out; training without one region and testing on that held-out region.

---

# Slides 18-21: Namare

## Slide 18 - Phase 4: Debugging the Initial Run Results

**5-line presentation script**

1. The first Phase 4 results looked promising because some runs showed HC-IoU around 0.66.
2. But after checking recall, precision, and predicted behavior, we found this was not true learning.
3. Some models predicted slum almost everywhere, while others predicted no slum at all.
4. We also found that formal-control FPR was missing and some socioeconomic ablation rows were misconfigured.
5. So the first result table became a debugging signal rather than a final scientific finding.

**Technical terms explained**

- **Class-prior behavior**: The model predicts based on class frequency instead of learning real features.
- **All-slum prediction**: A collapse where almost every pixel is predicted as slum.
- **No-slum prediction**: A collapse where almost no pixels are predicted as slum.
- **Balanced BCE**: Balanced Binary Cross-Entropy loss; a training loss adjusted so positive and negative classes contribute more equally.
- **Experiment registry**: A file that tracks which experiment rows are pending, running, done, or failed.

---

## Slide 19 - Where We Stopped and Why

**5-line presentation script**

1. This slide shows the corrected run after we added better diagnostics.
2. The visual-only model no longer hides behind a fake 0.66 score; its corrected HC-IoU is around 0.46.
3. Balanced accuracy is close to 0.50, which means visual features alone are weak and nearly random for this task.
4. The predicted-positive rate is no longer stuck at 0 or 1, so the collapse detector is working.
5. We paused here because continuing blindly would spend compute without enough evidence that the current labels support strong segmentation claims.

**Technical terms explained**

- **Corrected run**: A rerun after fixing training loss, metrics, and experiment configuration.
- **Nearly random**: Balanced accuracy close to 0.50 in a binary task means the model is not separating classes well.
- **Model collapse**: When a model falls into a trivial output pattern instead of learning meaningful discrimination.
- **Segmentation claim**: A claim that the model can classify individual pixels, not just whole regions or tiles.
- **Scientific gain**: Evidence that improves the validity of the research, not just more numbers.

---

## Slide 20 - Codebase Architecture and Future Roadmap

**5-line presentation script**

1. The codebase is organized around reproducibility and compute efficiency.
2. The `gee` folder handles Earth Engine exports, while `src/data` handles tiling, labels, and dataset loading.
3. LocateAnything feature extraction, model definitions, training, evaluation, and tracking are separated into their own modules.
4. The current pipeline works end-to-end, including caching, label verification, SAS-Net, and collapse diagnostics.
5. The next step is improving supervision and evaluation so the model can support stronger scientific claims.

**Technical terms explained**

- **Reproducibility**: The ability to rerun the experiment and get consistent behavior.
- **Preflight validation**: CPU-side checks before expensive GPU training.
- **Result recorder**: Code that writes per-run JSON and CSV results.
- **Orchestration**: The notebook logic that runs experiment phases in order.
- **Roadmap**: The planned future improvements needed to make the project stronger.

---

## Slide 21 - Closing Statement: The Path Forward

**5-line presentation script**

1. Our conclusion is not that BanglaSlumNet is finished, but that the experimental foundation is now solid.
2. We built an end-to-end cache-aware pipeline for Dhaka slum detection using VLM and socioeconomic fusion.
3. We identified and fixed major label, metric, and configuration bugs that produced misleading results.
4. The main lesson is that stronger supervision, or possibly tile-level framing, is needed before making definitive pixel-level claims.
5. The path forward is to add a manual evaluation subset, improve socioeconomic layers, and use high-resolution imagery for stronger VLM grounding.

**Technical terms explained**

- **Cache-aware pipeline**: A workflow that saves expensive outputs and reuses them automatically.
- **Configuration bug**: A setup error where the intended model or feature is not actually activated.
- **Tile-level framing**: Predicting labels for image patches rather than every pixel.
- **Manual evaluation subset**: A smaller set of human-checked labels used to fairly measure performance.
- **High-resolution imagery**: Images with smaller ground pixel size than Sentinel-2, useful for seeing narrow lanes and small rooftops.

---

# Quick Shared Glossary for the Whole Team

| Term | Simple explanation |
|---|---|
| BanglaSlumNet | Our proposed pipeline for Dhaka informal-settlement detection. |
| LocateAnything | NVIDIA vision-language grounding model used to produce prompt-based visual features. |
| VLM | Vision-Language Model; a model that connects images and text. |
| Sentinel-2 | Satellite imagery source used for the main optical tiles. |
| SAS-Net | Model used to normalize scene/appearance variation in image tiles. |
| Socioeconomic fusion | Combining visual features with context layers like nightlight and population. |
| VIIRS | Nighttime-light satellite data. |
| WorldPop | Population density dataset. |
| GHSL | Built-up and human settlement dataset. |
| Weak labels | Labels created from rules or proxies instead of manual annotation. |
| HC pixels | High-confidence pixels under the weak-label rule. |
| HC-IoU | Segmentation overlap score on high-confidence pixels. |
| Balanced accuracy | Metric that checks both positive and negative classes fairly. |
| Predicted-positive rate | Fraction of pixels the model predicts as slum. |
| FPR | False-positive rate; formal areas incorrectly predicted as slum. |
| Ablation | Experiment that removes/adds one component to measure its effect. |
| Cache | Saved intermediate result used to avoid recomputing. |
| Registry | File that tracks experiment progress and enables resume. |
| LORO | Leave-One-Region-Out generalization test. |
| Pixel-level segmentation | Predicting a class for every pixel. |
| Tile-level classification | Predicting a class for an entire image tile. |

