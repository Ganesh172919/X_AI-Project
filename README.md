# DS357 – Explainable AI (XAI) Project

## Research Replication and Extension

This project is part of the DS357 course focused on Explainable AI (XAI).

## Selected Paper
**InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly**  
(ICLR 2025) :contentReference[oaicite:0]{index=0}

## Problem Statement
SHAP is a powerful method for feature attribution in machine learning models.  
However, computing exact SHAP values is very slow and computationally expensive.  
This makes it difficult to use in real-time applications.

## Objective
- Understand the selected research paper deeply  
- Reproduce the original experiments  
- Identify limitations in the approach  
- Propose and implement an improved solution  
- Compare results with the original method  

## Dataset Used
- Standard tabular benchmark datasets  
- CUB (Caltech-UCSD Birds) dataset  

## XAI Method
- Original Method: Generalized Additive Models (GAM)  
- Proposed Method: Explainable Boosting Machine (EBM)  

## Key Idea
The original paper uses GAM to approximate SHAP values quickly.  
However, GAM assumes feature independence and ignores interactions.  

## Proposed Improvement
We replace GAM with EBM, which can model pairwise feature interactions.  
This improves explanation quality while maintaining interpretability.

## Implementation Details
- Model replication from the research paper  
- Implementation of XAI surrogate model  
- Training on the same datasets  
- Evaluation using accuracy and speed metrics  

## Expected Outcomes
- Faster SHAP value approximation  
- Better handling of feature interactions  
- Improved explanation accuracy  
- Real-time applicability  

## Project Phases
- Phase 1: Paper selection and proposal  
- Phase 2: Experiment replication  
- Phase 3: Research gap and extension  

## Repository Structure
- Phase_1_work: Paper analysis  
- Phase_2_work: Replication code  
- Phase_3_work: Improvement and results  

## Reproducibility
- Dataset links provided  
- Requirements file included  
- Instructions to run code  
- Random seed for consistency  

## Authors
Team of 5 students – DS357 Course Project