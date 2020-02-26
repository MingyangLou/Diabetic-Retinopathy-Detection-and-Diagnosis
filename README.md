# Diabetic Retinopathy Detection and Diagnosis
This project utilizes varaious machine learning models for Diabetic Retinopathy diagnosis and detection.

# Introduction
## What is Diabetic Retinopathy?
Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). At first, diabetic retinopathy may cause no symptoms or only mild vision problems. Eventually, it can cause blindness. (Refer: https://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611)

### Five stages of Diabetic Retinopathy:
   - Stage 0: No DR
   - Stage 1: Mild nonproliferative retinopathy
   - Stage 2: Moderate nonproliferative retinopathy 
   - Stage 3: Severe nonproliferative retinopathy
   - Stage 4: Proliferative diabetic retinopathy  
### Why not just rely on doctors?
  - Shortage of eye doctors
    
    Especially in underdeveloped areas
  - Poor eye doctors' exam consistency
    
    Only about 60%, according to a research performed by Google
    
    Eye doctors are good at detecting very healthy, and very unhealthy tissue, but have difficulty consistently diagnosing tissue that is in-between.
# Data Processing
- Data visualization 
- Image preprocessing
# Model Fitting
## Image Preprocessing
`Image_Preprocess.py`

We follow standard image processing procedures:
1. Rescale, Resize and RGB to grey
2. Data Augmentaion
3. Enhancement
## Modeling Fitting
`svm_lasso_lda.py`
- Use PCA/K means to extract image features
- Train machine learning models
  - SVM
  - Logistic Regression
  - LDA

`CNN.ipynb`
- CNN model (Use Google Colab)
# Model Evaluation
- 95% specificity
- 73% accuracy
