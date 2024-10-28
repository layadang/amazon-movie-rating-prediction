# Amazon Movie Review Ratings Prediction
### [📝 FINAL WRITE-UP HERE](https://github.com/layadang/amazon-movie-rating-prediction/blob/main/Laya_CS_506_Midterm_Write-up.pdf)

## About
This is the repo for my submission to CS 506 Midterm [Kaggle competition](https://www.kaggle.com/competitions/cs-506-midterm-fall-2024/) for highest accuracy in predicting star rating (out of 5) of Amazon Movie Reviews, given number of helpful votes (`HelpfulnessNumerator`), total votes (`HelpfulnessDenominator`), `Time`, `Text` (user review), and `Summary`. There are a total of 1,485,341 fully labeled data points to work with. No deep-learning (in models or libraries) is allowed.

## To recreate my final submission:
1. Create `data/` directory and upload `train.csv` and`test.csv` datasets (obtained from Kaggle competition)
2. Run `pip install requirements.txt` on a Python 3.10 environment
3. Run `feature_extraction.py`(about 40 minutes with multi-core processing) to get `data/full_features.csv` and `data/submission_features.csv`
4. Run `train_test_split.ipynb` (about 10 minutes) to get all data components for training, testing, and submission
5. Run `tfidf_ipynb` (about 10 minutes) to add TF-IDF matrix to all components
6. Run `xgb.ipynb` (about 20 minutes, excluding hyperparameter tuning) to create the model (`models/xgb_final.pkl`) and final submission

## Other files
* `helper.py`: Helper functions for resampling, SVD reduction, and creating submission files
* `data_exploration.ipynb`: Plots for EDA and final write-up
* `experiments/*`: Archive files for testing various models (keeping them for future reference)

## Thanks for reading!
🥉 Placed 3rd in final leaderboard
