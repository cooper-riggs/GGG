# Ghosts, Ghouls, and Goblins — Kaggle Classification Project

## Overview
This project predicts whether a candy belongs to a **Ghost, Ghoul, or Goblin** in the Kaggle *Ghosts, Ghouls, and Goblins* competition. The dataset includes categorical features such as color, ingredients, and other candy attributes. The goal was to build a full classification workflow in R and compare several models to produce accurate Kaggle submissions.

## What the Code Does
- Cleans and preprocesses the data using a tidymodels recipe  
  - Removes the `id` column  
  - Converts categorical predictors to factors  
  - Handles novel + rare categories  
  - Creates dummy variables  
  - Normalizes numeric predictors  
- Trains multiple classification models:
  - **Support Vector Machine (Linear Kernel)**  
  - **Random Forests (ranger)**  
  - **Naive Bayes**
- Tunes models using cross-validation and **accuracy**
- Fits the best workflow for each model type
- Generates predictions for the test set and writes **Kaggle-ready submission files**


