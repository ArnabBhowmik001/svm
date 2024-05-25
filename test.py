import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
# Assuming the data is saved in a CSV file named 'medical_data.csv'
# Load the data
data = pd.read_csv('tt.csv')

# print(data)
pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(data)
# Perform One-Hot Encoding for 'Urine Colour'
# encoder = OneHotEncoder(drop='first', sparse_output=False)
# encoded_urine_color = encoder.fit_transform(data[[' Urine Colour ']])
# encoded_df = pd.DataFrame(encoded_urine_color, columns=encoder.get_feature_names_out([' Urine Colour ']))
# print(encoded_df)
# Concatenate encoded columns with the original dataframe
# data = data.drop(' Urine Colour ', axis=1)
# data = pd.concat([data, encoded_df], axis=1)
data.fillna(0, inplace=True)
test_t=data[:1].drop([' Risk'], axis=1)
# test_t[' Urine Appearance ']=0;

# test_t=test_t.drop([' Light Yellow '], axis=1)
# Split the data into features and target
X = data.drop([' Risk'], axis=1)
y = data[' Risk']
# print(data.columns)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_test)
# Train a One-vs-Rest SVM model
model = OneVsRestClassifier(SVC())
model.fit(X_train, y_train.apply(str) )
model_pkl_file = "iris_classifier_model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)


with open(model_pkl_file, 'rb') as file:  
    model1 = pickle.load(file)

# evaluate model 
y_predict = model1.predict(X_test)

# check results

single_person_data = {
    'Total Leukocytes Count (WBC) ': 7200,
    ' Neutrophils ': 62,
    ' Lymphocyte ': 28,
    ' Monocytes ': 6,
    ' Eosinophils ': 3,
    ' Basophils ': 1,
    ' Immature Granulocyte Percentage ': 0.6,
    ' Total RBC ': 5.1,
    ' Nucleated Red Blood Cells ': 0,
    ' Nucleated Red Blood Cells % ': 0,
    ' Hemoglobin Hematocrit(PCV) ': 86,
    ' Mean Corpuscular Volume(MCV) ': 28.5,
    ' Mean Corpuscular Hemoglobin(MCH) ': 30.5,
    ' Mean Corp. Hemo. Conc (MCHC) ': 15.5,
    ' Red Cell Distribution Width - SD (RDW-SD) ': 13.5,
    ' Red Cell Distribution Width (RDW-CV) ': 13.5,
    ' Platelets ': 255500,
    ' Platelet Distribution Width (PDW) ': 16,
    ' Mean Platelet Volume (MPV) ': 10.5,
    ' Platelet Count ': 255000,
    ' Platelet to Large Cell Ratio (PLCR) ': 26.5,
    ' Plateletcrit (PCT) ': 0.22,
    ' Urine Volume ': 1.6,
    ' Urine Colour ':10,
    ' Urine Appearance ': 1,
    ' Urine Specific Gravity ': 1.018,
    ' Urine Crystals ': 0,
    ' Urine Bacteria ': 1,
    ' Urine Yeast ': 0,
    ' Urine Parasite ': 0,
    ' Urinary Leukocytes (PUS CELLS) ': 1,
    ' Urine Red Blood Cells ': 0,
    ' Urine Mucus ': 0,
    ' 25-OH Vitamin D (TOTAL) ':12,
    ' Vitamin B-12 ': 310,
    ' Total Cholesterol ': 195,
    ' HDL ': 48,
    ' LDL ': 123,
    ' Sodium ': 136,
    ' Potassium ': 4.4,
    ' Chloride ': 102,
    ' Total Triiodothyronine (T3) ': 1.4,
    ' Total Thyroxine (T4) ': 7.2,
    ' TSH - Ultrasensitive ': 1.01
}
single_person_df = pd.DataFrame([single_person_data])
# Evaluate the model
y_pred = model.predict(single_person_df)
print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)