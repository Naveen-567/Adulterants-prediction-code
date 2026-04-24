# NIR-Coupled Boosting Framework for Milk Adulterant Quantification

This repository contains the implementation of a high-precision soft sensor designed for the simultaneous quantification of common milk adulterants, including ammonia, urea, sugar, and hydrogen peroxide. 
The framework integrates near-infrared (NIR) spectroscopy with an ensemble learning approach featuring Orthogonal Partial Least Squares (OPLS) and the XGBoost algorithm.

---

## Project Overview

Milk adulteration is a critical global food safety concern. This study provides a rapid, non-destructive, and reliable method for detecting and quantifying multiple adulterants in real time, achieving an average error rate of less than 10% across diverse commercial milk matrices.

### Key Components
* **Soft Sensor:** An NIR-chemometric framework achieving high predictive performance ($CV-R^2$ up to 0.97 for urea).
* **Dimensionality Reduction:** Utilizes an updated OPLS algorithm to separate orthogonal noise from predictive variation, simplifying the model while increasing variance coverage.
* **Ensemble Learning:** Employs the XGBoost regression technique for robust target quantification, offering resistance to overfitting and high calculation speeds.
* **Feature Interpretation:** Integration of SHAP (Shapley Additive exPlanations) for wavelength importance ranking.

---

## Technical Architecture

The framework operates through an automated pipeline for preprocessing, spectral slicing, and regression.

### Hardware Specifications
* **System:** M1-powered MacBook Pro.
* **Memory:** 8 GB RAM.
* **Storage:** 256 GB SSD.

### Model Specifications
* **Preprocessing:** Savitzky-Golay filtering (second derivative, 11 filter windows, third-order polynomial) performed via the Nippy Python framework.
* **Spectral Slicing:** Automated NIR region selection for urea ($4500-5500~cm^{-1}$), ammonium sulfate ($7000-8000~cm^{-1}$), sugar ($9000-10000~cm^{-1}$), and $H_2O_2$ ($6100-6900~cm^{-1}$).
* **Optimization:** Hyperparameters fine-tuned via grid search (learning rate: 0.015, max depth: 2, estimators: 600).

---

## Getting Started

### Prerequisites
* **Python Version:** 3.6.
* **Libraries:** NumPy, Pandas, SciPy, scikit-learn, XGBoost, Nippy, and PyOPLS.

### Installation
```bash
git clone [https://github.com/Naveen-567/Adulterants-prediction-code](https://github.com/Naveen-567/Adulterants-prediction-code)
cd Adulterants-prediction-code
pip install numpy pandas scipy scikit-learn xgboost nippy pyopls
```

### Usage
Model Building: Utilize the 100 datasets (85% training, 15% testing) generated via Random K-means DOE.
Quantification: Input raw NIR spectra (4000-10000 $cm^{-1}$) into the Python core for real-time concentration analysis.
Validation: Thresholds for Limit of Detection (LOD) and Limit of Quantification (LOQ) are empirically established within the logic

### Citation
If you utilize this framework or code in your research, please cite:

Jesubalan, N. G., Chhabra, H., & Rathore, A. S. (2026). A Prediction Framework for Quantification of Milk Adulterants Using a NIR-Coupled Boosting Algorithm. Journal of Food Science, 91:e70966. doi:10.1111/1750-3841.70966




