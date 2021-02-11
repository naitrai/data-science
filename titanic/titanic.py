# Importing data
import pandas as pd
import numpy as np

# Import data
gender_sub = pd.read_csv('./gender_submission.csv')
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Exploratory data on train
fem_survived = train.loc[(train.Sex == 'female')]['Survived']
print("Number of women surviving: {}".format(sum(fem_survived)/len(fem_survived)))

male_survived = train.loc[(train.Sex == 'male')]['Survived']
print("Number of women surviving: {}".format(sum(male_survived)/len(male_survived)))

# Machine learning 
from sklearn.ensemble import RandomForestClassifier
y_train = train.loc[:,'Survived']

features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test.PassengerId,
    'Survived': predictions
})
output.to_csv('./my_submission.csv', index=False)
print("Your submission was successfully saved!")