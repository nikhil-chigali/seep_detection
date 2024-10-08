# SEEP Detection - Image Segmentation Task
## Project Overview
This project aims to develop and evaluate a deep convolutional neural network (DCNN) model for detecting and segmenting oil seeps in synthetic aperture radar (SAR) images. The dataset consists of SAR images, where each pixel is classified as either a non-seep or one of seven seep classes.

The primary objective is to segment the regions that contain seeps, with an optional task to classify these seeps into specific categories further.

## Data
The data consists of SAR images with corresponding masks. Each image is 256 x 256 pixels and is provided in `.tif` format. The images are located in two directories:

- train_images_256/ — Contains the SAR images.
- train_masks_256/ — Contains the segmentation masks for the images.

Each pixel in the masks is classified as:

- 0 - Non-seep
- 1-7 - Different classes of seeps
## Objective
**Segmentation**: Segment the regions in the SAR images that correspond to seeps.
Optional Classification: Further classify each segmented seep into one of the seven seep classes.

## Data preprocessing
The classes have been re-structured as follows skipping the `No Seep` class
- 0: Seep cls1
- 1: Seep cls2
- 2: Seep cls3
- 3: Seep cls4
- 4: Seep cls5
- 5: Seep cls6
- 6: Seep cls7

## Metrics
- **Training loss** 
  - **Intersection Over Union (IoU) or Jaccard Index Loss**: Measures the Intersection over Union (IoU) between predicted and ground truth masks.
- **Evaluation**
  - **Average Precision (AP)**: Computes the average precision across different IoU thresholds.
  - **Mean Average Precision (mAP)**: Assesses the quality of the instance masks generated by the model. It averages the AP across all classes and IoU thresholds.

## Training Results
> **Observation**: The model Segments and classifies `Seep cls1` well due to an abundance of its examples in the dataset. However, it performs poorly on other classes. This could be improved by weighting the loss function for less frequent classes.
### Confusion matrix across classes
![Confusion matrix](https://github.com/nikhil-chigali/seep_detection/blob/main/runs/segment/train/confusion_matrix_normalized.png)

Poor performance across all the classes except for `Seep cls1`. 

### Number of instances of each class
![Num instances](https://github.com/nikhil-chigali/seep_detection/blob/main/runs/segment/train/num_instances.png)

Skew in class distribution of the dataset

### Precision recall curve
![PRcurve](https://github.com/nikhil-chigali/seep_detection/blob/main/runs/segment/train/MaskPR_curve.png)

Poor PR curve - less area under the PR curve for every class, which mean the mAP is poor for each class.

### Training
![training](https://github.com/nikhil-chigali/seep_detection/blob/main/runs/segment/train/results.png)

- Steady decrease in Training and Val losses
- Mean Average Precision (mAP-M) for masks is steadily increasing until it performs well on `Seep cls1` then it seizes to improve. 
- Augmenting the data with more examples from `Seep cls2 - Seep cls7` can help the performance of the model on rest of the classes.

### Some prediction results
#### Labels
![labels](https://github.com/nikhil-chigali/seep_detection/blob/main/runs/segment/train/val_batch0_labels.jpg)

#### Predictions
![preds](https://github.com/nikhil-chigali/seep_detection/blob/main/runs/segment/train/val_batch0_pred.jpg)
