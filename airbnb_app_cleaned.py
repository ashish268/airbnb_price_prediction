import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


mydata = pd.read_csv('F:/AB_NYC_2019.csv')

mydata.head()

mydata.dtypes

mydata["last_review"] = pd.to_datetime(mydata["last_review"])
mydata.dtypes

mydata.isnull().sum()

mydata['reviews_per_month'].fillna(mydata['reviews_per_month'].mean(),inplace=True)
mydata.drop(columns=['host_name', 'last_review'], axis=1, inplace=True)
mydata.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Filter only numeric columns for correlation
numeric_cols = mydata.select_dtypes(include=['float64', 'int64']).columns
numeric_data = mydata[numeric_cols]

# Calculate correlation matrix with only numeric data
corr_1 = numeric_data.corr()

# Create the heatmap
fig, ax = plt.subplots(figsize=(8, 8))
dropSelf = np.zeros_like(corr_1)
dropSelf[np.triu_indices_from(dropSelf)] = True
sns.heatmap(corr_1, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()

plt.figure(figsize=(10,10))
ax = sns.violinplot(data=mydata, x="neighbourhood_group", y="availability_365")

plt.figure(figsize=(10,10))
sns.barplot(data=mydata, x='neighbourhood_group', y='price')

plt.figure(figsize=(10,6))
# Use x and y parameters instead of positional arguments
# This tells seaborn which columns to use for each axis
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', data=mydata)
plt.ioff()

def categorise(hotel_price):
    if hotel_price<=75:
        return 'Low'
    elif hotel_price >75 and hotel_price<=500:
        return 'Medium'
    else:
        return 'High'
mydata['price'].apply(categorise).value_counts().plot(kind='bar');

#word cloud

# First, install the wordcloud package

# Then import the required libraries
import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt  # Added matplotlib import which was missing

# Create the text from your data
text = " ".join(str(each) for each in mydata.name)

# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)

# Only need one figure definition - removed the duplicate
plt.figure(figsize=(15,10))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.figure(figsize=(8,8))
mydata['number_of_reviews'].plot(kind='hist')
plt.xlabel("Price")
plt.ioff()
plt.show()

mydata['name'].isnull().sum()

mydata['name'].fillna('', inplace=True)
mydata['name'].isnull().sum()

import re
def remove_punctuation_digits_specialchar(line):
    return re.sub('[^A-Za-z]+', ' ', line).lower()

mydata['clean_name'] = mydata['name'].apply(remove_punctuation_digits_specialchar)
# Let's compare raw and cleaned texts.
mydata[['name', 'clean_name']].head()

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load data (replace with your actual file)
mydata = pd.read_csv('F:/AB_NYC_2019.csv')

# Prepare stopwords
stop_words = set(stopwords.words('english'))

# Tokenization function
def tokenize_no_stopwords(line):
    # Check if the input is a string, if not return an empty string
    if not isinstance(line, str):
        return ""
    
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens if w.isalpha()]  # keep only words
    tokens_no_stop = [w for w in tokens if w not in stop_words]
    return " ".join(tokens_no_stop)

# Apply to dataset
# First, fill NaN values with empty strings to avoid the float error
mydata['name'] = mydata['name'].fillna("")
mydata['final_name'] = mydata['name'].apply(tokenize_no_stopwords)

print(mydata[['name', 'final_name']].head())

# First, install the lightgbm package

# Then run your original code
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report, mean_absolute_error, r2_score
from imblearn.over_sampling import RandomOverSampler

def classify_as_cheap_or_expensive(line):
    if line > 300:
        return 1
    else:
        return 0
        
mydata['target'] = mydata['price'].apply(classify_as_cheap_or_expensive)
mydata['target'].value_counts()
train, test = train_test_split(mydata, test_size=0.2, random_state=315, stratify=mydata['target'])

X_train, y_train = train['final_name'], train['target']
X_test, y_test = test['final_name'], test['target']
vect = TfidfVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)
ros = RandomOverSampler(sampling_strategy='minority', random_state=1)

# Update this line - fit_sample is deprecated, use fit_resample instead
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

lr = LGBMClassifier(random_state=315)
lr.fit(X_train_ros, y_train_ros)
preds = lr.predict(X_test)

print(classification_report(y_test, preds))
print("Accuracy: {0:.3f}".format(accuracy_score(y_test, preds)))
print("Recall: {0:.3f}".format(recall_score(y_test, preds)))

# Option 1: Drop only the columns that exist
# First check if columns exist, then drop them
columns_to_drop = []
if 'target' in mydata.columns:
    columns_to_drop.append('target')
if 'clean_name' in mydata.columns:
    columns_to_drop.append('clean_name')
    
if columns_to_drop:  # Only drop if there are columns to drop
    mydata.drop(columns=columns_to_drop, inplace=True)

# Option 2: Alternative approach using list comprehension
mydata.drop(columns=[col for col in ['target', 'clean_name'] if col in mydata.columns], 
            inplace=True)

mydata.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                                            # Fit label encoder
le.fit(mydata['neighbourhood_group'])
mydata['neighbourhood_group']=le.transform(mydata['neighbourhood_group'])    # Transform labels to normalized encoding.

le = LabelEncoder()
le.fit(mydata['neighbourhood'])
mydata['neighbourhood']=le.transform(mydata['neighbourhood'])

le =LabelEncoder()
le.fit(mydata['room_type'])
mydata['room_type']=le.transform(mydata['room_type'])


mydata.head()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split  # Added missing import
from sklearn.impute import SimpleImputer  # Added for handling NaN values
import numpy as np  # Added missing import

lm = LinearRegression()
mydata = mydata[mydata.price > 0]
mydata = mydata[mydata.availability_365 > 0]

X = mydata[['neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]

# Handle categorical variables - convert to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)  # Assuming pandas is imported as pd

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', 'most_frequent', or 'constant'
X = imputer.fit_transform(X)

# Prices are not normally distributed as well as there is alot of noise
y = np.log10(mydata['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error
y_predicts = lm.predict(X_test)

print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
        r2_score(y_test, y_predicts) * 100,
        mean_absolute_error(y_test, y_predicts)
        ))


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from sklearn import metrics

# Change 'mse' to 'squared_error' which is the new parameter name in newer scikit-learn versions
Reg_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=0)
Reg_tree = Reg_tree.fit(X_train, y_train)

y_predicts = Reg_tree.predict(X_test)
print("median absolute deviation (MAD): ", np.mean(abs(np.multiply(np.array(y_test.T-y_predicts), np.array(1/y_test)))))
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(metrics.mean_squared_error(y_test, y_predicts)),
        r2_score(y_test, y_predicts) * 100,
        mean_absolute_error(y_test, y_predicts)
        ))

mydata.head()

# First, install the fuzzywuzzy package

# Then import and use it
from fuzzywuzzy import process

def airbnb_finder(title):
    all_titles = mydata['final_name'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

title = airbnb_finder('village')
title



