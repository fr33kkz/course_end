import pickle
import pandas as pd

bank=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",na_values=["?",","])

bank.head(3)

bank.isnull().sum()

bank["Personal Loan"].value_counts()

bank.set_index(bank["ID"],inplace=True)

bank.head()

bank=bank.drop(["ID","ZIP Code"],axis=1)

bank.head(6)

bank.describe(include='all')

catcols = ['Education','CD Account','Online','CreditCard','Securities Account']
bank[catcols]=bank[catcols].astype('category')

bank=pd.get_dummies(bank, drop_first = True)
bank.dtypes

bank["Personal Loan"].value_counts(normalize= True)

from sklearn.model_selection import train_test_split

bank['Personal Loan']=bank['Personal Loan'].astype('category')
y=bank["Personal Loan"]
X=bank.drop('Personal Loan', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, )

from sklearn.preprocessing import StandardScaler
num_atr=X_train.select_dtypes(['int64','float64']).columns
num_atr

scaler = StandardScaler()
scaler.fit(X_train[num_atr])

X_train[num_atr]=scaler.transform(X_train[num_atr])
X_test[num_atr]=scaler.transform(X_test[num_atr])

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV


rc = RandomForestClassifier(n_jobs=-1, max_features='sqrt')

param_grid = {
           "n_estimators" : [5,10,30,40,50],
           "max_depth" : [5,6,7,8,10],
           "criterion": ["entropy","gini"]}

model = GridSearchCV(estimator=rc, param_grid=param_grid,cv= 3)
model.fit(X=X_train, y=y_train)

print('saving model as pkl file.......')
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
