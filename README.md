<p align="center">
  <img src="https://github.com/user-attachments/assets/d7917902-e39c-4807-b6db-08756c5fb970" alt="NeuroLens" width="100%">
</p>

**Authors:** Hrițcu Alexandru-Marius, Condrea Sebastian

### Contents

- [1. Introduction](#1-introduction)
- [2. Requirements and Specifications](#2-requirements-and-specifications)
- [3. Data Used and Its Organization](#3-data-used-and-its-organization)
- [4. Mathematical Model and Algorithms Used](mathematical_model.md)
- [5. Implementation Methodology](#5-implementation-methodology)
- [Preliminary Bibliography](#bibliografie)

---

## 1. Introduction


NeuroLens is an application for classifying brain tumors from MRI images using image processing and machine learning.


### 1.1. Problem Definition and Scope

The designed system, to be implemented, will act as an assignment function that maps the domain, formed by the space of images, to a value in the codomain of solutions, formed by the tumor types (or the possible absence of a tumor, a case which may nonetheless represent a system error, a result that must be studied in greater depth).

### 1.2. Purpose and Objectives

The purpose of this project is to implement a learning model that can produce a high-accuracy result (preferably $\geq 95\,\%$) within a useful time ($\leq 300\,\mathrm{ms}$), while at the same time making the program easy to use and keeping testing and training times sufficiently short.

---

## 2. Requirements and Specifications

The software must be able to run on a mid-range device with deterministic results.

### 2.1. Functional Requirements

In the first phase, for the first code demo, the program will be usable from the terminal by providing a reference to the image to be scanned. After the first demo and the verification of results, we will add a UI element to make the program easier to use. Additional system functions will come from the ability to import and export the system parameters, so that training can be done on a much more powerful device than the one used for testing.

#### 2.1.1. Image Loading and Validation

The targeted image formats are the most popular ones, but for the MVP we plan to support a single image format, namely JPG.

---

## 3. Data Used and Its Organization

The data used is largely taken from the dataset compiled by Masoud Nickparvar, and will be stored in the database of the model's training and validation program.

### 3.1. The Dataset

The dataset contains 1,400 images from each brain tumor category intended for training, and another 400 from each for testing the model, of which 10 of each type will be set aside to ensure a proper demo.

### 3.2. Data Preparation for the Model

To prepare the images and make the learning model's job easier, we plan to use various processing techniques (contrast, masking, etc.) and post-processing (segmentation).

### 3.3. Candidate Data Sources

The table below compares the public datasets analyzed (sources 6–9 in the [bibliography](#bibliografie)). The dataset chosen for the project is the one by Nickparvar [[6]](#ref6), because it includes all four target classes.

| No. | Author(s) / Year | Dataset Title | Application | Technologies Used | Methodology | Results | Limitations | Additional Comments |
|---|---|---|---|---|---|---|---|---|
| 6 | M. Nickparvar, 2025* | Brain Tumor MRI Dataset | Training and evaluating models for classifying brain tumors into four classes. | MRI images, image processing, CNN, Deep Learning. | Dataset combining images from Figshare, SARTAJ, and Br35H, organized for training and testing. | **7,023 MRI images**: glioma, meningioma, pituitary tumor, and no tumor. | Images of different sizes that require resizing; possible differences between sources. | Candidate for NeuroLens, as it includes the four target classes. |
| 7 | J. Cheng, 2024 (version 8) | brain tumor dataset | Classification and segmentation of brain tumors from MRI. | T1 contrast-enhanced MRI images, MATLAB, image processing. | Organization of the images into three tumor categories, with associated information and annotations. | **3,064 MRI images** from 233 patients: glioma, meningioma, and pituitary tumor. | Does not contain the no-tumor class; the distribution of images across classes is imbalanced. | Useful for evaluating three-class classification. |
| 8 | S. Bhuvaji et al., 2020* | Brain Tumor Classification (MRI) | Classification of MRI images into four classes. | MRI images, CNN, Transfer Learning, image processing. | Organized into Training and Testing directories, with subdirectories corresponding to the classes. | Approximately **3,264 files** for glioma, meningioma, pituitary tumor, and no tumor. | Small size; possible imbalances and overlaps with other public datasets. | Useful for initial experiments and comparisons. |
| 9 | A. Hamada, 2020 | Br35H :: Brain Tumor Detection 2020 | Detecting the presence or absence of a brain tumor in MRI. | MRI images, CNN, Deep Learning, binary classification. | Organization of the images into two categories: with tumor and without tumor. | Dataset for distinguishing normal images from those with tumors; not the result of any particular model. | Does not allow identification of the tumor type; not sufficient on its own for multiclass classification. | Can help complete the no-tumor class. |

---

## 4. Mathematical Model and Algorithms Used

This section provides the mathematical foundations for the image-processing operators, the custom convolutional network, the loss function, the optimization, and the evaluation metrics. Because of its size, it is in a separate file:

**[Mathematical Model and Algorithms Used](mathematical_model.md)**

---

## 5. Implementation Methodology

This section describes how the mathematical operators defined earlier are translated into distinct software modules. The inference pipeline will implement $\mathcal{P} \rightarrow \Phi_{\theta}\rightarrow U_c \mathbf v+\mathbf b_c\rightarrow\mathrm{softmax}\rightarrow\mathrm{arg\,max}$.

Unlike the ResNet-34 or YOLO models used in various articles on this subject, we will not load pretrained weights; the convolutional layers and the classifier will be initialized and optimized within the project.

### 5.1. Preparing Data for Training, Validation, and Testing

The data will be prepared in the following order: file inventory, structure validation, creation of a `manifest.json`, and assignment of images to splits. All learning activities, including statistical analysis, will be carried out exclusively on the training subset.

#### 5.1.1. Image Validation and Label Encoding

The received files will undergo a brief check of their integrity, dimensions, number of channels, and the presence of compatible labels, in order to avoid situations that lead to unexplained results in the trained model.

We will use the labels: glioma = 0; meningioma = 1; no tumor = 2; pituitary tumor = 3.

### 5.2. Implementing the Image Processing Pipeline

The preprocessing $\mathcal P$ will be implemented through composable operators. The order in which they are applied to the image matters!

#### 5.2.1. Resizing and Intensity Normalization

We will initially use a monochrome input of $128\times128$, a size also found in the from-scratch CNN experiment, where it was reduced from $512\times512$ to lower the computational cost [[2]](#ref2).

Resizing will use bilinear interpolation, and the aspect ratio will be preserved by padding the borders.

Pixel values will be converted to floating point and scaled to the interval $[0,1]$. The standardization rule with mean and standard deviation will be taken from the training data and applied to the test data.

#### 5.2.2. Filtering, Masking, and Image Segmentation

The baseline variant will not additionally alter anatomical structures. There will be controlled experiments with Gaussian filtering for high-frequency noise, median filtering for impulse noise, histogram equalization, and region masks. The article by Roy et al. describes the limitations of linear and nonlinear filters and, for Otsu, shows that global binarization can include unwanted regions of the MRI [[1]](#ref1).

If segmentation is tested, the Otsu threshold will be computed according to the between-class variance criterion from the mathematical model, after which morphological operations and connected-component analysis can be applied. The resulting mask will not be treated as a certified tumor segmentation. For pituitary tumors, aggressive cropping of extracranial regions may remove part of the region of interest.

#### 5.2.3. Applying Augmentations to the Training Data

Affine transformations will be used: small rotations, moderate translations and scalings, together with limited intensity variations. An initial configuration is rotation in the interval $[-10^\circ, 10^\circ]$, translation of up to 5% of the size, and scaling between $0.9$ and $1.1$. Interpolation and border padding will avoid uncontrolled loss of anatomical regions.

Each augmentation will be generated from the training images after the subsets have been separated. The same model, with the same split, will be compared with and without augmentation, to estimate whether the reduction of the augmented empirical risk translates into improved generalization. Reflections or strong deformations will not be used without checking anatomical plausibility. Sultan et al. applied flip, noise, and rotation as examples of augmentation [[2]](#ref2).

### 5.3. Implementing the Custom Convolutional Network

The classifier will be implemented using tensor operations, but the topology and all the parameters of the model will be defined by us. We will draw on research in this field to guide our architectural decisions, relying on the researchers' experiments to save time [[2]](#ref2), [[4]](#ref4).

#### 5.3.1. Layer Configuration and Parameter Initialization

A starting architecture will have 3 successive convolutional blocks of type $3\times3$ convolution, batch normalization, ReLU, with max-pooling aggregation after selected blocks. The initial number of filters may increase from 16 to 32 and 64. The final maps are reduced through spatial aggregation, followed by a linear layer with four outputs. This configuration is a proposal inspired by the organization of CNNs described in the article by Sultan et al. [[2]](#ref2).

The tensor dimensions for each layer will be computed before running, using the kernel width, stride, and padding defined mathematically. The weights will be initialized for ReLU activations, and the biases will receive controlled initial values. The network configuration and the number of trainable parameters will be saved together with the checkpoint.

#### 5.3.2. Implementing the Forward Pass and the Output for the Four Classes

Each processed image $X$ will be transformed into the feature vector $\mathbf v= \Phi_{\theta}(X)$. The last layer will produce $\mathbf z = U_c \mathbf v+ \mathbf{b}_c\in\mathbb R^4$, without an independent sigmoid activation for each class. For display, softmax is applied, and the label is the index of the maximum component, according to the justification of the $\mathrm{arg\,max}$ rule in the mathematical model.

During training, the cross-entropy will be computed directly from the scores $\mathbf z$ through a numerically stable implementation, avoiding redundant softmax–logarithm operations.

### 5.4. Training the Model

Training will adjust simultaneously the parameters of the extractor $\Phi_{\theta}$ and those of the final layer $U_c, \mathbf b_c$ using backpropagation over mini-batches. A log will be kept of the loss, the validation metrics, the epoch durations, and the configuration.

#### 5.4.1. Loss Computation and Parameter Updates

The loss function is the multiclass cross-entropy $-\sum_k t_{i,k}\log p_{i,k}$ described in the mathematical model. The gradient with respect to the scores is related to the difference between the predicted distribution and the indicator vector. For each mini-batch, the following will be executed: the forward pass, the loss computation, gradient reset, backpropagation, and the optimization step. We will start with Adam or AdamW, using the numerical stabilization and the gradient moments analyzed mathematically.

The $L_2$ penalty and dropout will be configurable, in training only, where appropriate. If the class frequencies differ greatly, the loss weighting will be considered based only on the distribution in the training set.

#### 5.4.2. Choosing Hyperparameters and Monitoring Validation

As a starting point, we will test mini-batches of 16 and 32 images, a learning rate on the order of $10^{-3}$–$10^{-4}$, and at most 30–40 epochs, depending on convergence and resources. These ranges are design decisions and will be adjusted only through validation, not by repeatedly consulting the test set. We will track the training and validation loss, the accuracy, and $F1_{\mathrm{macro}}$. A divergence between a decreasing training loss and stagnant validation performance will signal possible overfitting.

The preferred selection criterion will be maximizing the $F1_{\mathrm{macro}}$ score on validation. Early stopping will be applied after a fixed number of epochs without improvement.

#### 5.4.3. Saving the Model Selected on the Validation Set

Each time the validation criterion improves, a checkpoint will be saved that includes the weights, the layer configuration, the input size, the normalization parameters, the class order, the seed, and the number of epochs. To reproduce training, the optimizer state can be kept separately.

### 5.5. Evaluating the Model on the Test Set

A single final evaluation will be performed after the architecture and all hyperparameters have been frozen. In addition to accuracy, per-class metrics and $F1_{\mathrm{macro}}$ will be computed. The results will be interpreted with respect to the distribution of images and the risk of dependence between aggregates.

#### 5.5.1. Computing the Metrics and the Confusion Matrix

For each test example, $y_i$, $\widehat y_i$, and the four scores will be kept. The matrix $\mathrm{CM}\in\mathbb N^{4\times4}$ will use the convention already established: rows for the true label and columns for the prediction. From it, $TP_k$, $FP_k$, $FN_k$, and $TN_k$ will be derived in a *one-vs-rest* scheme, followed by precision, sensitivity, specificity, and $F1_k$; macro averages will also be reported.

#### 5.5.2. Analysis of Misclassifications

Errors will be grouped by the pair $(y_i,\widehat y_i)$. For each group, the image conditions will be inspected: orientation, contrast, edges, and resolution.

---
---

# 6 Arhitecture

## Package Diagram

<p align="center">
  <img src="https://github.com/user-attachments/assets/9e33547b-5dc8-4fc5-b963-96c22486339f" alt="Package Diagram" width="700">
</p>

## Sequence Diagram

<p align="center">
  <img src="https://github.com/user-attachments/assets/75644e8e-6ae1-45d4-9741-0d75f16da48d" alt="Sequence Diagram" width="700">
</p>

## Data Diagram

<p align="center">
  <img src="https://github.com/user-attachments/assets/33c5a9e1-d677-44b6-a39b-0a7b28ec0095" alt="Data Diagram" width="100%">
</p>

## Activity Diagram

<p align="center">
  <img src="https://github.com/user-attachments/assets/7c50744c-0ab4-48eb-bb1d-e8688a8641de" alt="Activity Diagram" width="400">
</p>

## Preliminary Class Diagram

<p align="center">
  <img src="https://github.com/user-attachments/assets/0b028e34-38be-4a9b-a1c4-8bf3ff121ceb" alt="Preliminary Class Diagram" width="100%">
</p>

<a id="bibliografie"></a>

## Preliminary Bibliography

<a id="ref1"></a>**[1]** S. Roy, S. Nag, I. K. Maitra and S. K. Bandyopadhyay, *A Review on Automated Brain Tumor Detection and Segmentation from MRI of Brain*, 2013. Pages used: **3–5** (preprocessing, median, mean, and Gaussian filtering) and **7–9** (thresholds, segmentation, the Otsu method). <https://arxiv.org/abs/1312.6150>

<a id="ref2"></a>**[2]** H. H. Sultan, N. M. Salem and W. Al-Atabany, *Multi-Classification of Brain Tumor Images Using Deep Neural Network*, IEEE Access, vol. 7, 2019, pp. 69215–69225, DOI: [10.1109/ACCESS.2019.2919122](https://doi.org/10.1109/ACCESS.2019.2919122). Pages used: **69216–69220** (convolutional representations, datasets, resizing, architecture, and augmentation) and **69221–69223** (training, results, and evaluation).

<a id="ref3"></a>**[3]** A. Raza, H. Ayub, J. A. Khan and collaborators, *A Hybrid Deep Learning-Based Approach for Brain Tumor Classification*, Electronics, vol. 11, no. 7, art. 1146, 2022, DOI: [10.3390/electronics11071146](https://doi.org/10.3390/electronics11071146). Item used: **Table 4 — General confusion matrix**, republished as a figure on ResearchGate; the figure page has no printed pagination. <https://www.researchgate.net/figure/General-confusion-matrix_tbl3_359740416>

<a id="ref4"></a>**[4]** A. Shahin, *Fine-tuned ResNet34 for efficient brain tumor classification*, Scientific Reports, vol. 15, art. 36910, 2025, DOI: [10.1038/s41598-025-20872-3](https://doi.org/10.1038/s41598-025-20872-3). Pages used from the publisher's PDF: **2–3** (data, preprocessing, augmentation), **3–5** (CNN layers and optimization), **6** (confusion matrix, classification metrics, and macro averages). <https://pmc.ncbi.nlm.nih.gov/articles/PMC12546736/>

<a id="ref5"></a>**[5]** A. M. Taha, S. A. Aly and M. F. Darwish, *Detecting Glioma, Meningioma, and Pituitary Tumors, and Normal Brain Tissues based on Yolov11 and Yolov8 Deep Learning Models*, arXiv:2504.00189, 2025. Pages used from the PDF: **2** (datasets), **2–3** (normalization and geometric augmentation), **3–5** (metrics, confusion matrix, and comparison with a custom CNN). <https://arxiv.org/pdf/2504.00189>

<a id="ref6"></a>**[6]** M. Nickparvar, *Brain Tumor MRI Dataset*, Kaggle. Items used: **the dataset description and the split into four classes**; web resource without numbered pages. The numerical distributions and the provenance of the collection were also verified in Shahin (2025), pp. 2–3. <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

<a id="ref7"></a>**[7]** J. Cheng, *brain tumor dataset*, Figshare, version 8, December 21, 2024, DOI: [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427). Items used: **the collection description, the version, the files, and the README**; data resource without pagination, with $3\,064$ contrast-enhanced T1 images for three classes. <https://figshare.com/articles/dataset/brain_tumor_dataset/1512427>

<a id="ref8"></a>**[8]** S. Bhuvaji, A. Kadam, P. Bhumkar and S. Dedge, *Brain Tumor Classification (MRI)*, Kaggle, DOI: [10.34740/KAGGLE/DSV/1183165](https://doi.org/10.34740/KAGGLE/DSV/1183165). Items used: **Description and the Training/Testing structure with the class subfolders**; data resource without pagination. <https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri>

<a id="ref9"></a>**[9]** A. Hamada, *Br35H :: Brain Tumor Detection 2020*, Kaggle. Items used: **the description and the binary organization into yes/no classes**; data resource without pagination. The structure and number of images are also discussed in the NeuroLens report, p. 5. <https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection>

### Comparative Analysis of the Sources

#### Scientific Articles (sources 1–5)

| No. | Author(s) / Year | Article Title | Application | Technologies Used | Methodology | Results | Limitations | Additional Comments |
|---|---|---|---|---|---|---|---|---|
| 1 | S. Roy et al., 2013 | A Review on Automated Brain Tumor Detection and Segmentation from MRI of Brain | Automatic detection and segmentation of brain tumors from MRI images. | Image processing, median filtering, segmentation thresholds, the Otsu method. | Comparative analysis of preprocessing, detection, and segmentation techniques for medical images. | Highlights the advantages and disadvantages of existing methods, without proposing a single classifier. | The performance of the methods can be affected by noise, contrast, and the similarity between tumor and healthy tissue. | Survey article useful for the theoretical grounding of MRI image preprocessing and segmentation. |
| 2 | H. H. Sultan et al., 2019 | Multi-Classification of Brain Tumor Images Using Deep Neural Network | Classification of brain tumors and differentiation of glioma grades. | Deep Learning, CNN, contrast-enhanced MRI images. | Training a convolutional network on two datasets, one for three tumor types and another for three glioma grades. | Reported accuracy of **96.13%** for tumor types and **98.7%** for glioma grades. | Relatively small datasets; generalization to images from other institutions is not demonstrated. | Demonstrates the use of CNNs for multiclass classification of MRI images. |
| 3 | A. Raza et al., 2022 | A Hybrid Deep Learning-Based Approach for Brain Tumor Classification | Classification of brain tumors into glioma, meningioma, and pituitary tumor. | DeepTumorNet, GoogLeNet, CNN, Transfer Learning, Leaky ReLU. | Modifying the GoogLeNet architecture by replacing the last five layers with fifteen new layers and training the model for three classes. | Reported accuracy of **99.67%**, precision 99.6%, sensitivity 100%, and F1 score of 99.66%. | More complex architecture; results depend on the dataset and require validation on independent data. | Benchmark for comparing the NeuroLens models. |
| 4 | A. Shahin, 2025 | Fine-tuned ResNet34 for efficient brain tumor classification | Classification of MRI images into four classes: glioma, meningioma, pituitary tumor, and no tumor. | ResNet34, CNN, Transfer Learning, Ranger Optimizer, RAdam, Lookahead, data augmentation. | Fine-tuning the pretrained ResNet34 model, introducing a custom classifier, and training on 7,023 MRI images. | Reported accuracy of **99.66%**, macro precision 99.67%, macro sensitivity 99.66%, and macro F1 99.66%. | The dataset size and variations between MRI scanners may affect generalization. | Relevant for NeuroLens because of the four classes and the pretrained model. |
| 5 | A. M. Taha et al., 2025 | Detecting Glioma, Meningioma, and Pituitary Tumors, and Normal Brain Tissues based on Yolov11 and Yolov8 Deep Learning Models | Classification of MRI images for three tumor types and normal brain tissue. | YOLOv8, YOLOv11, CNN, Transfer Learning, data augmentation. | Fine-tuning pretrained YOLO models for four categories and comparing with a custom CNN. | Reported accuracy of **99.49%** for YOLOv8, **99.56%** for YOLOv11, and **96.98%** for the custom CNN. | Need for validation on external datasets; the models may require high computational resources. | Allows comparison of YOLO with CNN and ResNet. |

Datasets (sources 6–9) are compared in [section 3](#33-candidate-data-sources).

