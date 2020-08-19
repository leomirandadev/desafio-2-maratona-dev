import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        return data.drop(labels=self.columns, axis='columns')

# read file
df_data_1 = pd.read_csv('dataset_desafio_2.csv')

# drop column name
rm_columns = DropColumns(columns=["NOME"])

rm_columns.fit(X=df_data_1)

df_data_2 = pd.DataFrame.from_records(
    data=rm_columns.transform(
        X=df_data_1
    ),
)

# remove nan values
si = SimpleImputer(
    missing_values=np.nan,
    strategy='constant',
    fill_value=0,
    verbose=0,
    copy=True
)

si.fit(X=df_data_2)

df_data_3 = pd.DataFrame.from_records(
    data=si.transform(
        X=df_data_2
    ), 
    columns=df_data_2.columns 
)

# change muito_bom tp excelente
si = SimpleImputer(
    missing_values='MUITO_BOM',
    strategy='constant',
    fill_value='EXCELENTE',
    verbose=0,
    copy=True
)

si.fit(X=df_data_3)

df_data_3 = pd.DataFrame.from_records(
    data=si.transform(
        X=df_data_3
    ), 
    columns=df_data_3.columns 
)


# # normalize grade
# col_names = ["NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO", "INGLES"]
# dataLines = df_data_3[col_names]
# scaler = StandardScaler().fit(dataLines.values)
# dataLines = scaler.transform(dataLines.values)
# df_data_3[col_names] = dataLines

# split predict and target variable
features = [
    "MATRICULA", 'REPROVACOES_DE', 'REPROVACOES_EM', "REPROVACOES_MF", "REPROVACOES_GO",
    "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO",
    "INGLES", "H_AULA_PRES", "TAREFAS_ONLINE", "FALTAS", 
]

target = ["PERFIL"]

x = df_data_3[features]
y = df_data_3[target]

# classifier test and training variables
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=np.random)

dtc_model = ExtraTreesClassifier(n_estimators=200)

dtc_model.fit(x_train, y_train.values.ravel())

# calculate accuracy
print "ACCURACY:"
accuracy = round(dtc_model.score(x_test, y_test), 2) * 100
print accuracy, "%"


