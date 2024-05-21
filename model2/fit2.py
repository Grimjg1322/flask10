import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

Dog_df=pd.read_excel("Dog.xlsx")
label_encoder = LabelEncoder()
Dog_df["Название породы"] = label_encoder.fit_transform(Dog_df["Название породы"])
X = Dog_df.drop(["Название породы"], axis=1)
Y = Dog_df["Название породы"]
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.3,random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
with open('Dogs.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)
