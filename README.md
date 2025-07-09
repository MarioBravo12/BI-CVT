
# Bi-CVT: Bimodal Convolutional Vision Transformer for Alzheimer's Disease Classification

## Project Overview

This repository contains the implementation of **Bi-CVT (Bimodal Convolutional Vision Transformer)**, a novel deep learning model for Alzheimer's Disease classification using multimodal data from the BrainLat dataset.

### Research Paper
**Title**: "Bi-CVT: An Interpretable Bimodal Convolutional Vision Transformer with Cross-Attention for EEG and Clinical Data Fusion in Alzheimer's Disease Classification"

**Performance**: 98.4% accuracy in distinguishing AD, bvFTD, and Healthy Controls

---

## Repository Structure

```
BI-CVT/
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── Modelo_BrainLat_EEG_Bimodal_Cross_Attention.ipynb # Main model implementation
├── Preprocesamiento_BrainLat_individual.ipynb        # EEG preprocessing pipeline
└── Gradcam_BrainLat_EEG_Bimodal_Cross_Attention.ipynb # Explainable AI analysis
```

---

## Dataset Information

### BrainLat Dataset
- **Source**: Latin American Brain Health Institute (BrainLat)
- **Total Participants**: 780 subjects from 5 Latin American countries
- **Age Range**: 21-89 years (Mean: 62.7 ± 9.5 years)
- **Reference**: Prado, P. et al. BrainLat-dataset. Synapse https://doi.org/10.7303/syn51549340 (2023)

### Diagnostic Groups
| Group | Description | Count |
|-------|-------------|-------|
| **AD** | Alzheimer's Disease | 278 patients |
| **bvFTD** | Behavioral variant Frontotemporal Dementia | 163 patients |
| **HC** | Healthy Controls | 250 subjects |
| **PD** | Parkinson's Disease | 57 patients |
| **MS** | Multiple Sclerosis | 32 patients |

### Data Modalities
- **EEG**: High-density resting-state EEG (128 channels)
- **Clinical**: Neuropsychological assessments and demographics
- **MRI**: Anatomical and functional MRI (available but not used in this study)
- **DWI**: Diffusion-weighted imaging (available but not used in this study)

---

## Model Architecture

### Bi-CVT Components

#### 1. EEG Processing Branch
- **Input**: 224×224×128 STFT spectrograms
- **Architecture**: Convolutional layers with residual connections
- **Purpose**: Extract spectral-temporal patterns from EEG signals

#### 2. Clinical Data Branch
- **Input**: Neuropsychological and sociodemographic features
- **Architecture**: 1D CNN + Dense layers
- **Purpose**: Process structured clinical assessments

#### 3. Cross-Attention Mechanism
- **Innovation**: Bidirectional attention between EEG and clinical modalities
- **Implementation**: Multi-Head Attention with learnable query-key-value mappings
- **Purpose**: Enable dynamic feature interaction and fusion

#### 4. Classification Head
- **Input**: Fused multimodal representations
- **Output**: 3-class probability distribution (AD/bvFTD/HC)

---

## Installation and Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Google Colab or Jupyter Notebook environment

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `tensorflow>=2.14.0` - Deep learning framework
- `mne>=1.2.0` - EEG data processing
- `spkit>=0.0.9.7` - Signal processing toolkit
- `opencv-python>=4.6.0` - Image processing
- `scikit-learn>=1.1.0` - Machine learning utilities
- `matplotlib>=3.5.0` - Visualization
- `seaborn>=0.11.0` - Statistical plotting
- `pandas>=1.4.0` - Data manipulation

---

## Usage Instructions

### 1. Data Preprocessing
Run the preprocessing notebook to convert raw EEG to time-frequency representations:

```bash
jupyter notebook Preprocesamiento_BrainLat_individual.ipynb
```

**Pipeline**:
- Load raw EEG files (.set format)
- Apply Short-Time Fourier Transform (STFT)
- Generate 224×224×128 spectrograms
- Save processed data as .npy files

### 2. Model Training
Execute the main model notebook for training and evaluation:

```bash
jupyter notebook Modelo_BrainLat_EEG_Bimodal_Cross_Attention.ipynb
```

**Process**:
- Load multimodal data (EEG + clinical)
- Build Bi-CVT architecture
- Train with cross-validation
- Evaluate performance metrics
- Save trained model

### 3. Explainable AI Analysis
Run the interpretability notebook for model explanation:

```bash
jupyter notebook Gradcam_BrainLat_EEG_Bimodal_Cross_Attention.ipynb
```

**Techniques**:
- Grad-CAM for EEG spectrograms
- Integrated Gradients for clinical features
- Attention weight visualization
- Cross-modal interaction analysis

---

## Expected Results

### Performance Metrics
- **Accuracy**: 98.4%
- **Precision**: >95% for all classes
- **Recall**: >95% for all classes
- **F1-Score**: >95% for all classes
- **AUC**: >0.98 for all classes

### Model Outputs
- Trained model weights (.hdf5 format)
- Performance metrics and confusion matrices
- Grad-CAM heatmaps for interpretability
- Attention weight visualizations
- Feature importance scores

---

## Methodology

### Data Processing
1. **EEG Preprocessing**:
   - STFT with 4-second windows, 75% overlap
   - Frequency range: 0-40 Hz (Delta, Theta, Alpha, Beta)
   - Spatial normalization across channels

2. **Clinical Data Processing**:
   - Standard scaling (z-score normalization)
   - Feature selection based on clinical relevance
   - Missing value imputation

3. **Multimodal Fusion**:
   - Cross-attention mechanism for dynamic interaction
   - Late fusion strategy with learned weights
   - Joint optimization of both modalities

### Model Training
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical crossentropy
- **Regularization**: Dropout, BatchNormalization, Early Stopping
- **Validation**: Stratified 5-fold cross-validation
- **Hardware**: NVIDIA GPU with CUDA support

---

## Evaluation Methods

### Quantitative Metrics
- **Classification Accuracy**: Overall correctness
- **Per-class Precision/Recall**: Class-specific performance
- **Confusion Matrix**: Detailed classification results
- **ROC Curves**: Threshold-independent evaluation
- **Cohen's Kappa**: Inter-rater agreement measure

### Qualitative Analysis
- **Grad-CAM**: Visual explanation of important EEG regions
- **Integrated Gradients**: Clinical feature importance
- **Attention Maps**: Cross-modal interaction patterns
- **Clinical Interpretation**: Neurological relevance assessment

---

## Clinical Significance

### EEG Biomarkers
- **Delta Waves (1-4 Hz)**: Associated with cognitive impairment
- **Theta Waves (4-8 Hz)**: Memory and attention processes
- **Alpha Waves (8-13 Hz)**: Resting state and alertness
- **Beta Waves (13-30 Hz)**: Active cognitive processing

### Clinical Features
- **Neuropsychological Tests**: MOCA, cognitive assessments
- **Demographics**: Age, education, socioeconomic factors
- **Functional Measures**: Daily living activities
- **Biomarker Integration**: Multi-domain assessment

---

## Reproducibility

### Random Seeds
All random processes use fixed seeds for reproducibility:
- NumPy: `np.random.seed(45)`
- TensorFlow: `tf.random.set_seed(45)`
- Python: `random.seed(45)`

### Environment
- Jupyter notebooks with markdown documentation
- Detailed comments explaining each step
- Version-controlled dependencies
- Google Colab compatibility

### Data Availability
- BrainLat dataset: Available through Synapse platform
- Preprocessing scripts: Included in repository
- Model weights: Available upon request

## Drive Data Repository

- **Drive Link**: [BrainLat Organized Dataset](https://drive.google.com/drive/folders/1IqmjsQvSktzwN2UIpfxrazXDHpNcEupa?usp=sharing)

This Google Drive folder contains the preprocessed and organized data files required to run the experiments, including:
- EEG spectrograms (STFT representations)
- Clinical feature matrices
- Train/test splits
- Sample visualization outputs

## References

1. **Primary Dataset**: Prado, P. et al. (2023). BrainLat-dataset. Synapse. https://doi.org/10.7303/syn51549340

2. **Attention Mechanisms**: Vaswani, A. et al. (2017). Attention Is All You Need. NIPS.

3. **Vision Transformers**: Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words. ICLR.

4. **Grad-CAM**: Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.

5. **MNE-Python**: Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience.

---

## Contributing

### Code Standards
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings
- Add unit tests for critical functions
- Document all hyperparameters

### Issue Reporting
- Use GitHub Issues for bug reports
- Include system information and error logs
- Provide minimal reproducible examples

---

## Contact

For questions about the implementation or requests for collaboration:

- **Primary Contact**: Mario Alejandro Bravo Ortiz
- **Institution**: Universidad Autonoma de Manizales
- **Email**: mario.bravoo@autonoma.edu.co

---

## Acknowledgments

- **BrainLat Consortium**: For providing the multimodal dataset
- **Latin American Brain Health Institute**: For dataset coordination
- **Research Participants**: For their valuable contribution to neuroscience research
- **Open Source Community**: For the tools and libraries that made this work possible

---

## Limitations

### Technical Limitations
- **Computational Requirements**: High-end GPU needed for training
- **Memory Usage**: ~16GB RAM recommended for full dataset
- **Processing Time**: ~6-8 hours for complete pipeline

### Clinical Limitations
- **Population Specificity**: Trained on Latin American populations
- **Diagnostic Scope**: Limited to AD, bvFTD, and HC classification
- **External Validation**: Requires testing on additional datasets

### Methodological Considerations
- **Cross-modal Dependency**: Performance depends on both modalities
- **Feature Selection**: Clinical features may need domain-specific tuning
- **Generalizability**: Model performance may vary across different populations

---

**Last Updated**: July 2025
**Version**: 1.0.0  
- **Reference**: Prado, P. et al. *BrainLat-dataset*. Synapse (2023).
- **Population**: 780 Latin American individuals (530 patients, 250 controls).
- **Modalities used**: High-density EEG and clinical/neuropsychological data.
- **Access**: Public, via Synapse DOI above (subject to registration).

## Repository Structure

```bash
.
├── Preprocesamiento_BrainLat_individual.ipynb      # Converts raw EEG signals into STFT spectrograms
├── Modelo_BrainLat_EEG_Bimodal_Cross_Attention.ipynb  # Bi-CVT model implementation (training + evaluation)
├── Gradcam_BrainLat_EEG_Bimodal_Cross_Attention.ipynb # Visualization and interpretability (Grad-CAM, IG, attention)
└── README.md                                        # This file
```

## Requirements

To run the code, install the following dependencies (e.g. via `pip install -r requirements.txt`):

```text
python==3.9
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
tensorflow>=2.10
keras
librosa
torch
captum
```

## Usage Instructions

1. **Preprocess EEG Data:**
   - Run `Preprocesamiento_BrainLat_individual.ipynb`
   - This converts EEG into spectrograms using Short-Time Fourier Transform (STFT).
   - Output: numpy arrays or images of shape `(channels, time, frequency)`.

2. **Train and Evaluate the Bi-CVT Model:**
   - Run `Modelo_BrainLat_EEG_Bimodal_Cross_Attention.ipynb`
   - Input EEG spectrograms and tabular clinical data.
   - Evaluates the model using Accuracy, Precision, Recall, F1-Score.
   - Performs a **3-class classification** (AD, bvFTD, CN).

3. **Model Interpretability:**
   - Run `Gradcam_BrainLat_EEG_Bimodal_Cross_Attention.ipynb`
   - Generates:
     - Grad-CAM visualizations on EEG attention maps.
     - Integrated Gradients on clinical features.
     - Cross-Attention interaction visualization.

## Methodology

- **EEG Processing**: Signals are transformed into 2D spectrograms using STFT.
- **Clinical Encoding**: Structured data is passed through 1D-CNN layers.
- **Fusion Module**: Cross-Attention allows both modalities to refine each other's representations.
- **Model**: Bi-CVT (CNN + Vision Transformer + Cross-Attention).
- **Evaluation**:
  - Train/test split stratified by class.
  - 5-fold cross-validation.
  - Metrics reported: accuracy, per-class F1, precision, recall, and balanced accuracy.
- **Ablation Study**: Performed to assess the impact of each modality and fusion mechanism.

## Evaluation

The model was evaluated on the BrainLat dataset with the following metrics:

- **Overall Accuracy**: 98.4%
- **F1-Score (Macro)**: 98.1%
- **Balanced Accuracy**: 97.9%
- **Ablation results**: Provided in the article, showing drop in performance when removing EEG, clinical data, or the cross-attention module.

## Limitations
- Model was trained on a balanced subset of BrainLat; real-world imbalance needs further testing.

- Model interpretability still requires validation by domain experts.

- EEG variability across recording centers may affect generalization.

## Citation

If you use this code or dataset, please cite:

```
Mario Alejandro Bravo-Ortiz, Guevara-Navarro, E., & Holguin García, S. A. (2025). 
MarioBravo12/BI-CVT: BI-CVT First Version (v1.0.1). Zenodo. 
[https://doi.org/10.5281/zenodo.15722443](https://doi.org/10.5281/zenodo.15849991)
```

And the dataset:

```
Prado, P. et al. (2023). BrainLat-dataset. Synapse. https://doi.org/10.7303/syn51549340
```
