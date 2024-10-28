import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

print("====LOADING DATA AND PACKAGES====")

full_data = pd.read_csv("./data/train.csv")
submission = pd.read_csv("./data/test.csv")

submission = pd.merge(full_data, submission, on=['Id']).drop(columns=['Score_x'], axis=1).rename(columns={'Score_y': 'Score'})
full_data = full_data.dropna(subset=['Score'])

# full_data = pd.read_csv("./data/full_features.csv")
# submission = pd.read_csv("./data/submission_features.csv")

nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words = stop_words + ['film', 'movie', 'one', 'way', 'first', 
                           'get', 'time', 'one', 'like', 'new', 'many', 
                           'think', 'made', 'make', 'know', 'see', 
                           'films', 'fact', 'really', 'story', 'since', 
                           'around', 'second', 'old', 'take', 'actually', 
                           'going'] # least important words according to xgb

# tfidf_vectorizer = TfidfVectorizer(max_df=0.55, min_df=0.01, max_features=500)

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

print("====COMPLETED LOADING DATA & SET-UP====")

def remove_stopwords(text):
    """ 
        also puts words in lowercase
    """
    new_text = []
    for word in text.split():
        if word.lower() not in stop_words:
            new_text.append(word.lower())
    return ' '.join(new_text)

def lemmatize_text(data):
    lemmatized_data = [
        " ".join(lemmatizer.lemmatize(word) for sent in sent_tokenize(message) for word in word_tokenize(sent))
        if message else ""
        for message in data
    ]
    return lemmatized_data

def extract_features(df, isSubmission=False):

    print("====BEGAN FEATURE EXTRACTION====")
    print({df.shape})

    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')

    # can't remove rows from submissions!
    if not isSubmission: 
        print(f"Before filtering: {df.shape}")
        # remove errors where numerator is larger than denominator for helpfulness
        df = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]
        print(f"After filtering: {df.shape}")

    print("====STARTING BASIC FEATURE EXTRACTION====")

    # Convert UNIX date to readable date
    df['Date'] = pd.to_datetime(df['Time'], unit='s')
    print({df.shape})

    # Get year and month
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    print({df.shape})

    # Get helpful and unhelpful 
    df['HelpfulnessNumerator'] = df['HelpfulnessNumerator'].fillna(0)
    df['Helpful'] = df['HelpfulnessNumerator']
    df['Unhelpful'] = df['HelpfulnessDenominator'] - df['HelpfulnessNumerator']
    print({df.shape})

    # clean summary
    print("====STARTING TEXT FEATURE EXTRACTION====")
    # punctuation
    df['ExclaimationCount'] = df['Text'].parallel_apply(lambda x: x.count('!'))
    df['QuestionCount'] = df['Text'].parallel_apply(lambda x: x.count('?'))
    df['AllCapsCount'] = df['Text'].parallel_apply(lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1))

    # word count and uniqueness
    df['Words'] = df['Text'].parallel_apply(lambda x: ' '.join(re.findall(r'\b[a-zA-Z]+\b', x)))     # clean text
    df['WordCount'] = df['Words'].parallel_apply(lambda x: len(x.split()) if pd.notna(x) else 0)     # word count
    print({df.shape})

    print("Word counts complete")

    df['TextCleaned'] = df['Words'].parallel_apply(remove_stopwords)  # remove stopwords, if that makes text empty, take the summary
    df['TextCleaned'] = df.parallel_apply(lambda row: row['Summary'] if not row['TextCleaned'] else row['TextCleaned'], axis=1)
    print({df.shape})

    print("Removing stopwords complete")

    df['UniqueWords'] = df['Words'].parallel_apply(lambda words: len(set(words.split())))
    df['UniqueWords'] = df['UniqueWords'] / df['WordCount']
    df['UniqueWords'] = df['UniqueWords'].fillna(0)
    print({df.shape})

    # print("Unique words complete")

    df['Summary'] = df['Summary'].parallel_apply(lambda x: ' '.join(re.findall(r'\b[a-zA-Z]+\b', x.lower())) if pd.notna(x) else '')
    print({df.shape})

    print("Summary processing complete")

    ####### TF-IDF & SENTIMENT STUFF #######
    # doing it on summary right now bc its faster

    print("====STARTING SENTIMENT EXTRACTION====")

    df['LemmatizedSummary'] = lemmatize_text(df['Summary'])
    df['LemmatizedCleanedText'] = lemmatize_text(df['TextCleaned'])

    print("Lemmatizing complete")
    print({df.shape})
    
    df['SummarySentiment'] = df['LemmatizedSummary'].parallel_apply(lambda x: sia.polarity_scores(x)['compound'])
    print({df.shape})
    
    df['CleanedTextSentiment'] = df['LemmatizedCleanedText'].parallel_apply(lambda x: sia.polarity_scores(x)['compound'])
    print({df.shape})

    print("Sentiment complete")

    # print("====STARTING TF-IDF====")

    # x = tfidf_vectorizer.fit_transform(df['LemmatizedSummary'])
    # tfidf_df = pd.DataFrame(x.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    # result = pd.concat([df, tfidf_df], axis=1)

    df = df.drop(columns=['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Date', 'Time'])

    return df

print("====FULL DATASET EXTRACTION BEGIN====")
full_features_df = extract_features(full_data)
full_features_df.to_csv("data/full_features.csv", index=False, header=True)
print("====FULL DATASET EXTRACTION END====")

print("====SUBMISSION FILE EXTRACTION BEGIN====")
submission_features_df = extract_features(submission, isSubmission=True)
submission_features_df.to_csv("data/submission_features.csv", index=False, header=True)
print("====SUBMISSION FILE EXTRACTION END====")

print("====FILES SAVED====")