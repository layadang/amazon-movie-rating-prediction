import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD

# RESAMPLING FUNCTION
def resample_data(size, X_train, y_train):
    """ 
        Resample the training data to size (0.5 to half the data, 0.25 to quarter it, etc.)
    """

    # recombine training data
    X_train['Score'] = y_train

    desired_ratio = {
        1: 0.06,
        2: 0.06,
        3: 0.11,
        4: 0.22,
        5: 0.55
    }
    
    ## a little less skewed
    # desired_ratio = {
    #     1: 0.10,
    #     2: 0.10,
    #     3: 0.20,
    #     4: 0.25,
    #     5: 0.35
    # }
    ## equal
    # desired_ratio = {
    #     1: 0.2,
    #     2: 0.2,
    #     3: 0.2,
    #     4: 0.2,
    #     5: 0.2
    # }

    # half the total training size
    total_size = size * len(X_train)
    balanced_data = pd.DataFrame()

    # loop through each Score category and resample it
    for score, ratio in desired_ratio.items():

        # determine how many samples we need for this score category
        target_count = int(total_size * ratio)
        
        # get all the rows that belong to the current Score category
        score_data = X_train[X_train['Score'] == score]
        
        if len(score_data) > target_count:
            # if the class is overrepresented, perform undersampling (probably this one since we're downsizing)
            score_resampled = resample(score_data, replace=False, n_samples=target_count, random_state=123)
        else:
            score_resampled = resample(score_data, replace=True, n_samples=target_count, random_state=123)
        
        # append the resampled data to the balanced_data DataFrame
        balanced_data = pd.concat([balanced_data, score_resampled])
        
    balanced_data = balanced_data.sample(frac=1, random_state=123).reset_index(drop=True)

    # reassign
    y_train = balanced_data['Score']
    X_train = balanced_data.drop(columns=['Score'], axis=1)

    print(f"new training size is {balanced_data.shape}")

    return X_train, y_train

# REDUCING TF-IDF BY SVD FUNCTION
def svd_reduce_and_combine(X_train, start=0, end=500, components=250):
    """ 
        end is last index of column to reduce (500 by default)
        components is the svd deconstruction lim (250 by default)
    """
    X_train_tfidf = X_train.iloc[:, start:end]
    X_train_non_tfidf = pd.concat([X_train.iloc[:, :start], X_train.iloc[:, end:]], axis=1)

    svd = TruncatedSVD(n_components=components, random_state=123)
    X_train_tfidf_reduced = svd.fit_transform(X_train_tfidf)

    X_train_reduced = np.hstack((X_train_tfidf_reduced, X_train_non_tfidf.values))

    columns = [f'SVD_{i+1}' for i in range(components)] + list(X_train_non_tfidf.columns)
    X_train_reduced = pd.DataFrame(X_train_reduced, columns=columns)

    return X_train_reduced

# MAKE SUBMISSION CSV FUNCTION
submission = pd.read_csv("data/test.csv")
def get_submission_ready(predictions):
    submission_ready = pd.DataFrame({
            'Id': submission['Id'],
            'Score': predictions
        })
    submission_ready.to_csv("./data/submission.csv", index=False)
    print('Saved as submission.csv success!')