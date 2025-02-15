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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score


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
    atrs_n = ['arrival_date_year', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'company', 'agent']#, 'total_guests']
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
    ('imputer', SimpleImputer(strategy="mean")),
    ('mean', StandardScaler()),#std_scaler#Standarization
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
    ('imputer', SimpleImputer(strategy="mean")),
    ('mean', StandardScaler()),#std_scaler#Standarization
    ])

    num_attribs = X_test_num#For get numeric data
    cat_attribs = X_test_cat#For category data
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ]) 
    
    X_test = full_pipeline.fit_transform(X_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    log_reg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=500,
                   multi_class='multinomial', n_jobs=-1, penalty='l2',
                   random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
    log_reg.fit(X_train, y_train)
    y_predict_lr = log_reg.predict(X_test)
    y_test_lr = y_test
    X_test_lr = X_test
    print(classification_report(y_test, y_predict_lr))
    print(accuracy_score(y_test, y_predict_lr))
