import pandas as pd                                         # For data manipulation
import matplotlib.pyplot as plt                             # For plotting graphs
import seaborn as sns                                       # For visualizations
from sklearn.model_selection import train_test_split        # To split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier         # For Random Forest classification
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score  # Evaluation metrics
from sklearn.metrics import accuracy_score                  # For accuracy calculation
from sklearn.preprocessing import StandardScaler            # For feature scaling

# Load the dataset from a CSV file
data = pd.read_csv("C:/Python312/injury_data.csv")

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())


# Define features (X) and target variable (y)
X = data[['Player_Age','Player_Weight','Player_Height','Previous_Injuries','Training_Intensity','Recovery_Time']]  # Features
y = data['Likelihood_of_Injury']  # Target

# Check if target variable is binary (should contain only 0 and 1)
print("\nUnique values in the target column:")
print(y.unique())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced') 
model.fit(X_train, y_train) 

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
print('\n--------------------------Confusion Matrix:---------------------------------------------')
print(cm)

report = classification_report(y_test, y_predict)
print('----------------------------Classification Report:----------------------------------------')
print(report)

accuracy = accuracy_score(y_test, y_predict)
print('----------------------------Prediction Report:-------------------------------------------')
print('Accuracy:           ', accuracy)

# Calculate ROC AUC Score
if len(y.unique()) == 2: 
    prediction = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) 
    print('ROC AUC Score:      ', prediction)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# Function to test the model with custom input
def test_model():
   
    player_age = float(input("Enter Player Age: "))
    player_weight = float(input("Enter Player Weight (kg): "))
    player_height = float(input("Enter Player Height (cm): "))
    previous_injuries = int(input("Enter Number of Previous Injuries: "))
    training_intensity = float(input("Enter Training Intensity (scale 1-10): "))
    recovery_time = float(input("Enter Recovery Time (days): "))

    input_data = [[player_age, player_weight, player_height, previous_injuries, training_intensity, recovery_time]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  

    print("\nPredicted Likelihood of Injury: ", "Yes" if prediction == 1 else "No")
    print("Injury Probability: {:.2f}%".format(probability * 100))

# Call the function to test with user input
test_model()
