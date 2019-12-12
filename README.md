# Diabetic Retinopathy Diagnosis and Detection
This project utilizes varaious machine learning models for Diabetic Retinopathy diagnosis and detection.

# Content
1. [Introduction](#introduction)
2. Data Processing
3. Model Fitting
    1. SVM (Support Vector Machine)
    2. LASSO
    3. FFN
	      - RBM (restricted Boltzmann machine)
        - MMF 
        - NNMF(Non-negative matrix factorization)
    4. CNN (Hyperparameter tuning)
4. Model Evaluation
5. Conclusion

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
