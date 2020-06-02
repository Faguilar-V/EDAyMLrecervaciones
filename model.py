#Missing values
from sklearn.impute import SimpleImputer
#For standarization
from sklearn.preprocessing import StandardScaler
#For work encode categorical atrubuts
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#For do a best a work flow
from sklearn.pipeline import Pipeline
#Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# For search best parameters for ours models
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
#for evaluate our models
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__': 
    ########################################################################
    #Load our data sets
    train_set = pd.read_csv('files/train.csv')
    test_set = pd.read_csv('files/test.csv')
    
    #New atributes
    
    #Train
    train_set['total_guests'] = train_set['adults'] + train_set['children'] + train_set['babies']
    train_set['total_days'] = train_set['stays_in_week_nights'] + train_set['stays_in_weekend_nights']
    #Test
    test_set['total_guests'] = test_set['adults'] + test_set['children'] + test_set['babies']
    test_set['total_days'] = test_set['stays_in_week_nights'] + test_set['stays_in_weekend_nights']
    
    #Preparing data for our model
    
    #Train
    X_train = train_set.drop('is_canceled', axis=1)
    y_train = train_set['is_canceled'].copy()
    #Train
    X_test = test_set.drop('is_canceled', axis=1)
    y_test = test_set['is_canceled'].copy()
    
    #Cleaning data
    
    #Numerical atributs droped
    atrs_n = ['arrival_date_year', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'company', 'agent', 'total_guests']
    #Categorical atributs droped 
    atrs_cat = ['reservation_status', 'reservation_status_date', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type']
    atrs = atrs_cat + atrs_n
    #Train
    X_train = X_train.drop(atrs, axis=1)
    #Test
    X_test = X_test.drop(atrs, axis=1)
    
    #Encoding category type data
    
    #########
    #Train
    X_train_num = X_train.select_dtypes(exclude=['object', 'category']).columns
    X_train_cat = X_train.select_dtypes(include=['object', 'category']).columns
    num_pipeline = Pipeline([
    #('imputer', SimpleImputer(strategy="constant")),
    #('attribs_adder', CombinedAttributesAdder()), #Experimenting with atributes combinations
    ('std', StandardScaler()),#std_scaler#Standarization
    ])

    num_attribs = X_train_num#For get numeric data
    cat_attribs = X_train_cat#For category data
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ]) 
    X_train = full_pipeline.fit_transform(X_train)
    #########
    #Test
    X_test_num = X_test.select_dtypes(exclude=['object', 'category']).columns
    X_test_cat = X_test.select_dtypes(include=['object', 'category']).columns
    num_pipeline = Pipeline([
    #('imputer', SimpleImputer(strategy="constant")),
    #('attribs_adder', CombinedAttributesAdder()), #Experimenting with atributes combinations
    ('std', StandardScaler()),#std_scaler#Standarization
    ])

    num_attribs = X_test_num#For get numeric data
    cat_attribs = X_test_cat#For category data
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ]) 
    
    X_test = full_pipeline.fit_transform(X_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    ########################################################################    
    
    # define models to test:
    base_models = [("DT_model", DecisionTreeClassifier(random_state=42)),
                   ("RF_model", RandomForestClassifier(random_state=42,n_jobs=-1)),
                   ("LR_model", LogisticRegression(random_state=42,n_jobs=-1)),
                   ("svm_model", svm.SVC(random_state=42)),
                  ]
    #For 1 model first
    param_grid = {'n_estimators': [200],
                  'max_features': ['auto'],
                  'criterion' :['entropy', 'gini']
                 }
    forest_reg = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=forest_reg, param_grid=param_grid, cv=3, verbose=True)
    
    grid_search.fit(X_train, y_train)
    #For evaluate model
    final_model = grid_search.best_estimator_
    y_predict = final_model.predict(X_test)
    print(classification_report(y_test, y_predict))

