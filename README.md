# ISLES (ISCHEMIC STROKE LESION SEGMENTATION)

### Visual Result
<div align="center">
 <img src="./assets/result.fig.gif">
 <br>
 <em align="center">Fig 1: Segmentation on SISS dataset.</em>
</div>

### 1) About
**The purpose of this project is to build a CNN model for stroke lesion segmentaion using ISLES 2015 dataset.**
<p align="justify">Recent studies have shown the potential of using magnetic resonance imaging (MRI) in diagnosing ischemic stroke. Reviewing hundreds of slices produced by MRI, however, takes a lot of time and can lead to numerous human errors. It is widely accepted by the medical practitioners that automated segmentation methods for ischemic stroke lesions could significantly speed up the beginning of a patient’s treatment. The automated segmentation can locate the tissue with lesions and give an estimate of its volume, which helps in the clinical practice by providing a better assessement and evaluation of the risks of each treatment. These reasons highlight the need for a fully automatic ischemic stroke lesion segmentation approach using a flexible, fast and effective deep neural network.</p>

### 2) Dataset
### 2.1) SISS Dataset

*File:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;There are 4 types of MRI scan for one person*

*File Format:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
.nii*

*Image Shape:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
240(Slide Width) × 240(Slide Height) × 155(Number of Slide) × 4(Multi-mode)*

*Image Mode:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
4 (Multi-mode)*

<br>
<div align="center">
  <img src="./assets/siss.jpg">
 <br>
 <em align="center">Fig 2:SISS dataset.</em>
</div>


### 2.2) SPES Dataset

*File:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;There are 7 types of MRI scan for one person*

*File Format:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
.nii*

*Image Shape:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
96(Slide Width) × 110(Slide Height) × 71(Number of Slide) × 7(Multi-mode)*

*Image Mode:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
7 (Multi-mode)*

<br>
<div align="center">
  <img src="./assets/spes.jpg">
 <br>
 <em align="center">Fig 3: SPES dataset.</em>
</div>

### 2.3) Data Augmentation
|    Methods    |     Range     |
| ------------- | ------------- |
|   rescale     |   1.0/255     |
|rotation_range |    30         |
|horizontal_flip|    True       |
|vertical_flip  |    True       |
|shear_range    |    0.2        |
|zoom_range     |    0.1        |

### 3) My CNN Architecture
### 4) Evaluation Metric
True Positive (TP): TP implies number of true positives, that is, positive correctly identified as positive.
<br>
True Negative (TN): TN implies number of true negatives, that is, negative correctly identified as negative.
<br>
False Positive (FP): FP implies number of false positives, that is, negative incorrectly identified as positive.
<br>
False Negative (FN): FN implies number of false negatives, that is, positive incorrectly identified as negative.
#### 1. Dice Similiarty Coefficient:
![DSC(Dice Similiarty Coefficient)=\frac{2N_{TP}}{2N_{TP}+ N_{FP}+N_{FN}}](https://latex.codecogs.com/svg.latex?%5C%20DSC%3D%5Cfrac%7B2N_%7BTP%7D%7D%7B2N_%7BTP%7D&plus;%20N_%7BFP%7D&plus;N_%7BFN%7D%7D)
<br>
#### 2. Sensitivity
![Sensitivity=\frac{N_{TP}}{N_{TP}+ N_{FN}}](https://latex.codecogs.com/svg.latex?SEN%3D%5Cfrac%7BN_%7BTP%7D%7D%7BN_%7BTP%7D&plus;%20N_%7BFN%7D%7D)
<br>
#### 3. Specificity
![Specificity=\frac{N_{TN}}{N_{TN}+ N_{FP}}](https://latex.codecogs.com/svg.latex?SPC%3D%5Cfrac%7BN_%7BTN%7D%7D%7BN_%7BTN%7D&plus;%20N_%7BFP%7D%7D)

### 5) Ortimizer and Hyperparameter
### 5.1) Optimizer 
[Adam Optimizer](https://arxiv.org/pdf/1412.6980.pdf)
### 5.2) Hyperparameter
![LearningRate=Lr_i * f^{(epoch / step)}](https://latex.codecogs.com/svg.latex?LearningRate%3DLr_i%20*%20f%5E%7B%28epoch%20/%20step%29%7D)
<br>
Lri = Initial Learning Rate = 0.0001
<br>
decay factor(f) = 0.2
<br>
step = 2

### 5) Results
<div align="center">
 <img src="./assets/result.png">
 <br>
 <em align="center">Fig 6:Performance of proposed network in term of dice coefficient on each modality on SISS and SPES Dataset for
various loss functions.</em>
 <br>
 <img src="./assets/siss-plot.png">
 <br>
 <em align="center">Fig 7:From left to right: Plot for loss DSC and accuracy for training and validation set on SISS dataset.
</em>
 <img src="./assets/spes-plot.png">
 <br>
 <em align="center">Fig 8:From left to right: Plot for loss DSC and accuracy for training and validation set on SPES dataset.</em>
</div>




