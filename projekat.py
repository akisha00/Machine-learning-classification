import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, accuracy_score, recall_score
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

plt.style.use('dark_background')

def object_to_cat(x):
    
    for col in x.select_dtypes(['object']):
        x[col] = x[col].astype("category")

def label_encode(x):
    
    for col in x.select_dtypes(['category']):
        x[col] = x[col].cat.codes


def grid_search(X,y):

    hyper_param = {
        "criterion": ['gini', 'entropy', 'log_loss'],
        "max_depth" : np.arange(5, 20, 5),
        "min_samples_split": np.arange(2,10,2),
        "min_samples_leaf": np.arange(2,10,2),
        "max_features": np.arange(2, 8, 2),
        "max_leaf_nodes": np.arange(2, 10, 2),
        "ccp_alpha" : np.arange(0,0.5,0.05)
    }

    grid = GridSearchCV(RandomForestClassifier(), hyper_param, cv = 5, n_jobs = 4)
    grid.fit(X, y)
    print(grid.best_params_)

    return grid.best_params_


def call_models(X_train, X_test, y_train, y_test, model ,call_num = 0):

    tree_model = clone(model)

    tree_model.fit(X_train, y_train)

    print("Unakrsna validacija:",cross_val_score(tree_model, X_train, y_train, cv = 5))
    predictions = tree_model.predict(X_test)


    print("Tačnost ", model.__class__.__name__, " pre podešavanja hiperparametara: ", "%.2f" % accuracy_score(y_test,predictions))
    print("Odziv", model.__class__.__name__, " nakon podešavanja hiperparametara: ", "%.2f" % recall_score(y_test,predictions))
    print("Preciznost", model.__class__.__name__, " pre podešavanja hiperparametara: ", "%.2f" % precision_score(y_test,predictions))
    print("F1 score", model.__class__.__name__, " pre podešavanja hiperparametara: ", "%.2f" % f1_score(y_test,predictions))

    print("\n")



    #hypers = grid_search(X_train, y_train)
    if model.__class__.__name__ == "RandomForestClassifier":
        if call_num == 1:
            hypers = {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 10, 'max_features': 6, 'max_leaf_nodes': 8, 'min_samples_leaf': 2, 'min_samples_split': 4}
            
        elif call_num == 2:
            hypers = {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 2, 'max_leaf_nodes': 8, 'min_samples_leaf': 2, 'min_samples_split': 8}
            
        elif call_num == 3:
            hypers = {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 5, 'max_features': 4, 'max_leaf_nodes': 8, 'min_samples_leaf': 6, 'min_samples_split': 8}
        

        grid_model = clone(model)

        grid_model.ccp_alpha = hypers['ccp_alpha']
        grid_model.criterion = hypers['criterion']
        grid_model.max_depth = hypers['max_depth']
        grid_model.max_leaf_nodes = hypers['max_leaf_nodes']
        grid_model.min_samples_split = hypers['min_samples_split']

        grid_model.fit(X_train, y_train)

        print("Unakrsna validacija:", cross_val_score(grid_model, X_train, y_train, cv = 5))
        grid_predictions = grid_model.predict(X_test)

        print("Tačnost", model.__class__.__name__, " modela nakon podešavanja hiperparametara: ","%.2f" % accuracy_score(y_test,grid_predictions))
        print("Odziv", model.__class__.__name__, " nakon podešavanja hiperparametara: ", "%.2f" % recall_score(y_test,grid_predictions))
        print("Preciznost", model.__class__.__name__, " nakon podešavanja hiperparametara: ", "%.2f" % precision_score(y_test,grid_predictions))
        print("F1 score", model.__class__.__name__, " nakon podešavanja hiperparametara: ", "%.2f" % f1_score(y_test,grid_predictions))

        fig, ax = plt.subplots(1,2)
        
        cnf_mtrx_grid = confusion_matrix(y_test, grid_predictions)

        ax[1].set_title(label = f'Matrica konfuzije {model.__class__.__name__} nakon podešavanja hiperparametara')
        cnf_mtrx_dsp_grid = ConfusionMatrixDisplay(cnf_mtrx_grid, display_labels=[False, True]).plot(ax = ax[1])

        cnf_mtrx = confusion_matrix(y_test, predictions)


        ax[0].set_title(label = f'Matrica konfuzije {model.__class__.__name__} pre podešavanja hiperparametara')
        cnf_mtrx_dsp = ConfusionMatrixDisplay(cnf_mtrx, display_labels=[False, True]).plot(ax = ax[0])

    else:
        fig, ax = plt.subplots()

        cnf_mtrx = confusion_matrix(y_test, predictions)
        ax.set_title(label = f'Matrica konfuzije {model.__class__.__name__}')
        cnf_mtrx_dsp = ConfusionMatrixDisplay(cnf_mtrx, display_labels=[False, True]).plot(ax = ax)


    fig.set_figheight(8)
    fig.set_figwidth(18)


    plt.show(block=False)

    if model.__class__.__name__ == "RandomForestClassifier":
        return [tree_model, grid_model]
    else:
        return tree_model


data = pd.read_csv("Employee.csv")

data.head()
data.describe()

data.notnull().sum()
data.duplicated().value_counts()

data.drop_duplicates(keep = 'first',inplace = True)
data = data.reset_index(drop=True)

data_encoded = data.copy()

fire_rate_per_year = np.array(data[["JoiningYear", "LeaveOrNot"]].groupby("JoiningYear").mean())[:,0]
fire_rate_per_city =  np.array(data[["City", "LeaveOrNot"]].groupby("City").mean())[:,0]
fire_rate_per_gender = np.array(data[["Gender", "LeaveOrNot"]].groupby("Gender").mean())[:,0]
fire_rate_per_edu_tier = np.array(data[["Education", "LeaveOrNot"]].groupby("Education").mean())[:,0]
fire_rate_per_age = np.array(data[["Age", "LeaveOrNot"]].groupby("Age").mean())[:,0]
fire_rate_per_exp_lvl = np.array(data[["ExperienceInCurrentDomain", "LeaveOrNot"]].groupby("ExperienceInCurrentDomain").mean())[:,0]

data[["City", "LeaveOrNot"]].groupby("City").mean()
cities = ["Bangalore", "New Delhi", "Pune"]

data[["Gender", "LeaveOrNot"]].groupby("Gender").mean()
genders = ["Female","Male"]

data[["Education", "LeaveOrNot"]].groupby("Education").mean()
educations = ["Bachelors", "Masters", "PHD"]


fig, ax = plt.subplots(2,3)

fig.set_figheight(12)
fig.set_figwidth(15)

ax[0,0].bar(data["JoiningYear"].unique(),fire_rate_per_year*100)
ax[0,0].set_title("Procenat otkaza po godini dolaska u firmu")
ax[0,0].set_ylabel("Procenat")
ax[0,0].set_xlabel("Godina")

ax[0,1].bar(cities,fire_rate_per_city*100, color="orange")
ax[0,1].set_title("Procenat otkaza po mestu stanovanja")
ax[0,1].set_ylabel("Procenat")
ax[0,1].set_xlabel("Mesto")

ax[0,2].bar(genders,fire_rate_per_gender*100, color="green")
ax[0,2].set_title("Procenat otkaza po polu radnika")
ax[0,2].set_ylabel("Procenat")
ax[0,2].set_xlabel("Pol")

ax[1,0].bar(educations,fire_rate_per_edu_tier*100, color="red")
ax[1,0].set_title("Procenat otkaza po stepenu obrazovanja")
ax[1,0].set_ylabel("Procenat")
ax[1,0].set_xlabel("Stepen obrazovanja")


ax[1,1].bar(data["Age"].unique(),fire_rate_per_age*100, color="purple")
ax[1,1].set_title("Procenat otkaza po starosti")
ax[1,1].set_ylabel("Procenat")
ax[1,1].set_xlabel("Starost")

ax[1,2].bar(data["ExperienceInCurrentDomain"].unique(),fire_rate_per_exp_lvl*100, color="firebrick")
ax[1,2].set_title("Procenat otkaza po godinama iskustva\n" + "u odredjenoj tehnologiji")
ax[1,2].set_ylabel("Procenat")
ax[1,2].set_xlabel("Godine iskustva")

plt.show(block=False)


object_to_cat(data_encoded)
label_encode(data_encoded)

X = data_encoded.drop(["LeaveOrNot"], axis = 1)
y = data_encoded["LeaveOrNot"]


X1_train, X1_test, y_train, y_test = train_test_split(X,y,stratify = y)
#{'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 10, 'max_features': 6, 'max_leaf_nodes': 8, 'min_samples_leaf': 2, 'min_samples_split': 4}
[tree_1, grid_1] = call_models(X1_train, X1_test, y_train, y_test, call_num = 1, model = RandomForestClassifier())


_ = call_models(X1_train, X1_test, y_train, y_test, call_num = 1, model = KNeighborsClassifier())



fig, ax = plt.subplots()

fig.set_figheight(8)
fig.set_figwidth(18)

ax.matshow(data_encoded.corr())
ax.set_title("Matrica korelacija")
ax.set_xticks(np.arange(len(data_encoded.columns)))
ax.set_yticks(np.arange(len(data_encoded.columns)))
ax.set_xticklabels(data_encoded.columns, fontdict=None, minor = False, rotation = "vertical")
ax.set_yticklabels(data_encoded.columns, fontdict=None, minor = False)

xd = np.array(data_encoded.corr())

for (i, j), z in np.ndenumerate(xd):
    ax.text(j ,i , '{:0.2f}'.format(z) , ha = "center", va = "center")


plt.show(block=False)


categorical_features = [0,2,3,4,5,6]

mut_inf =  mutual_info_classif(X,y, discrete_features=categorical_features, n_neighbors=5)

for i in range(len(X.columns)):
    print(X.columns[i], ": ","%.2f"% mut_inf[i])


X2_train = X1_train.drop(["Age", "EverBenched", "ExperienceInCurrentDomain"], axis = 1)
X2_test = X1_test.drop(["Age", "EverBenched", "ExperienceInCurrentDomain"], axis = 1)

tree_2, grid_2 = call_models(X2_train, X2_test, y_train, y_test, call_num = 2, model = RandomForestClassifier())
#{'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 2, 'max_leaf_nodes': 8, 'min_samples_leaf': 2, 'min_samples_split': 8}


select_k_class = SelectKBest(k = 4)
select_k_class.fit(X,y)
column_idx = select_k_class.get_support(indices=True)
X3_train = X1_train.iloc[:, column_idx]
X3_test = X1_test.iloc[:, column_idx]
X3_train


tree_3,grid_3 = call_models(X3_train, X3_test, y_train, y_test, call_num = 3, model = RandomForestClassifier())
#{'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 5, 'max_features': 4, 'max_leaf_nodes': 8, 'min_samples_leaf': 6, 'min_samples_split': 8}



pred1 = tree_1.predict(X1_test)
pred2 = tree_2.predict(X2_test)

print("Tacnosti svih modela")
print("Obican model:", "%.2f" % accuracy_score(y_test,  pred1))
print("Model sa izbacenim atributima koristeci pd.corr() i mutual_info_classif():", "%.2f" % accuracy_score(y_test, pred2))
print("Model sa izbacenim atributima koristeci SelectKBest():", "%.2f" % accuracy_score(y_test, tree_3.predict(X3_test)))

plt.show()