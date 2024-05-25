import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming the data is saved in a CSV file named 'medical_data.csv'
# Load the data
data = pd.read_csv('tt.csv')
pd.options.display.max_columns = None
# Perform One-Hot Encoding for 'Urine Colour'
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_urine_color = encoder.fit_transform(data[[' Urine Colour ']])
encoded_df = pd.DataFrame(encoded_urine_color, columns=encoder.get_feature_names_out([' Urine Colour ']))

# print(encoded_df)
# Concatenate encoded columns with the original dataframe
data = data.drop(' Urine Colour ', axis=1)
data = pd.concat([data, encoded_df], axis=1)
data.fillna(0, inplace=True) 
test_t=data[:1].drop([' Risk'], axis=1)
# test_t[' Urine Appearance ']=0;

# test_t=test_t.drop([' Light Yellow '], axis=1)
# Split the data into features and target
X = data.drop([' Risk'], axis=1)
y = data[' Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_test)
# Train a One-vs-Rest SVM model
model = OneVsRestClassifier(SVC())
model.fit(X_train, y_train.apply(str) )

# print(test_t)
# Evaluate the model
y_pred = model.predict(test_t)
# accuracy = accuracy_score(y_test, y_pred)
print(y_pred[0])

# # Optionally, you can print the classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
