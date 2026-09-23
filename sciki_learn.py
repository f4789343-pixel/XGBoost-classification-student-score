from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
import pandas as pd

df = pd.read_csv('student-mat.csv',sep=';')

features = [
    "age", "Medu", "Fedu",
    "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout",
    "Dalc", "Walc", "health",
    "absences"]

X = df[features]
y = (df["G3"] >= 10).astype(int).values

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = XGBClassifier(n_estimators=50,learning_rate=0.1,max_depth=0,reg_lambda=1,gamma=0)
model.fit(x_train,y_train)
predictions = model.predict(x_test)

cm = confusion_matrix(y_test,predictions)
print('Accuracy:',accuracy_score(y_test,predictions))
print('Precision:', precision_score(y_test,predictions))
print('recall:', recall_score(y_test,predictions))
print('Confusion Matrix:', cm)