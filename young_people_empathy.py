import pandas as pd
from sklearn.model_selection import train_test_split
import BaslineClassifier
import importlib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import svm


class YoungPeopleEmpathy:
    
    def loadData():
        train_file_path = "young-people-survey/responses.csv"
        train_data = pd.read_csv(train_file_path)
        return train_data
    
    def run_baseline_classifier(train_data):
        train_data =  train_data.dropna(subset=set(["Empathy"]))
        X = train_data.drop(["Empathy"],axis=1)

        Y = train_data["Empathy"]
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y,random_state = 0)

        train_Y.fillna(train_Y.mode()[0], inplace=True)
        test_Y.fillna(test_Y.mode()[0], inplace=True)

        h = BaslineClassifier.BaselineClassifier({})
        h.train(train_X,train_Y)
        print("Train Accuracy")
        predicted_labels_train = h.predictAll(train_X)
        print(accuracy_score(train_Y, predicted_labels_train))
        print("Test Accuracy")
        predicted_labels = h.predictAll(test_X)
        print(accuracy_score(test_Y, predicted_labels))
        
    
    def removeRowsHavingMissingValuesinCategories(train_data,category_list):
        
        train_data_reduced = train_data.dropna(subset=set(category_list))
        return train_data_reduced
    
    def getCategoricalColumns(train_data,category_list):
        train_data_category = train_data[category_list]
        return train_data_category
    
    def getListofNumericColumnsAndCategorical(X):
        cols = X.columns
        num_cols = X._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        return num_cols,cat_cols
    
    def getBMI(train_data_demog):
        
        bmi = pd.DataFrame(train_data_demog['Weight']/((train_data_demog['Height'])**2))
        bmi['bmi_index'] = ['underweight' if x < 18.5  else 'overweight' if x > 24.9 else 'healthy' for x in                bmi[0]]
        return bmi['bmi_index']
    
    # splitting data into train and test
    def splitDataTrainTest(train_data):
        num_cols,cat_cols = YoungPeopleEmpathy.getListofNumericColumnsAndCategorical(train_data)
        train_data_red = YoungPeopleEmpathy.removeRowsHavingMissingValuesinCategories(train_data,cat_cols)
        X = train_data_red.drop(["Empathy"],axis=1)
        Y = train_data_red["Empathy"]
        Y_num = pd.to_numeric(Y, downcast='signed')
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y_num,random_state = 0,test_size=0.2)
        num_cols = list(num_cols)
        num_cols.remove("Empathy")
        num_cols.remove('Height')
        num_cols.remove('Weight')
        return train_X, test_X, train_Y, test_Y,num_cols,cat_cols
    
    
    def fillNumericMissingValues(train_X,test_X):
        train_X_numeric = train_X._get_numeric_data()
        test_X_numeric = test_X._get_numeric_data()

        train_X_numeric.fillna(train_X_numeric.mean(), inplace=True)
        test_X_numeric.fillna(test_X_numeric.mean(), inplace=True)
        return train_X_numeric,test_X_numeric
    
    def encodeCategoricalData(train_X,test_X,cat_cols):
        train_X_cat = YoungPeopleEmpathy.getCategoricalColumns(train_X,cat_cols)
        test_X_cat = YoungPeopleEmpathy.getCategoricalColumns(test_X,cat_cols)
        train_X_cat_encoded = pd.get_dummies(train_X_cat)
        test_X_cat_encoded = pd.get_dummies(test_X_cat)
        return train_X_cat_encoded,test_X_cat_encoded
    
    def calculateBMI(train_X_numeric,test_X_numeric):
        train_data_dem = train_X_numeric[['Height','Weight']]
        train_data_dem['Height'] = train_data_dem['Height']/100
        #train_data_dem.head()
        test_data_dem = test_X_numeric[['Height','Weight']]
        test_data_dem['Height'] = test_data_dem['Height']/100

        bmi_index_train =  YoungPeopleEmpathy.getBMI(train_data_dem)
        bmi_index_train.columns = ['bmi_index']
        bmi_index_test =  YoungPeopleEmpathy.getBMI(test_data_dem)
        bmi_index_test.columns = ['bmi_index']

        train_X_numeric = train_X_numeric.drop(['Height','Weight'],axis=1)
        test_X_numeric = test_X_numeric.drop(['Height','Weight'],axis=1)

        bmi_index_train_encoded = pd.get_dummies(bmi_index_train)
        bmi_index_test_encoded = pd.get_dummies(bmi_index_test)

        train_X_numeric = train_X_numeric.join(bmi_index_train_encoded)
        test_X_numeric = test_X_numeric.join(bmi_index_test_encoded)
        return train_X_numeric,test_X_numeric
    
    def getRangeData(train_data_mix_n,test_data_mix_n,num_cols):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for num_col in num_cols:
            labels = ["{0} - {1}-{2} ".format(i, i + 2,num_col) for i in range(0, 6, 2)]
            df_train[num_col] = pd.cut(train_data_mix_n[num_col], range(0, 7, 2), right=False, labels=labels)

        for num_col in num_cols:
            labels = ["{0} - {1}-{2} ".format(i, i + 2,num_col) for i in range(0, 6, 2)]
            df_test[num_col] = pd.cut(test_data_mix_n[num_col], range(0, 7, 2), right=False, labels=labels)

        df_encoded_train = pd.get_dummies(df_train)
        df_encoded_test = pd.get_dummies(df_test)

        return df_encoded_train,df_encoded_test
    
    
    # Create our function which stores the feature rankings to the ranks dictionary
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x,2), ranks)
        return dict(zip(names, ranks))
    
    
    # Create empty dictionary to store the mean value calculated from all the scores
    def calculateRankMatrix(train_data_mix,train_Y,colnames,threshold):
        ranks = {} 

        clf = ExtraTreesClassifier()
        clf = clf.fit(train_data_mix, train_Y)


        ranks["tree"] = YoungPeopleEmpathy.ranking(clf.feature_importances_, colnames);

        xgb = XGBClassifier(max_depth = 10, min_child_weight = 8,gamma = 0.7,colsample_bytree = 0.7, subsample =            0.7,reg_alpha = 1)


        xgb = xgb.fit(train_data_mix, train_Y)


        ranks["xgb"] = YoungPeopleEmpathy.ranking(xgb.feature_importances_, colnames);

        ada = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1)
        ada = ada.fit(train_data_mix, train_Y)

        ranks["ada"] = YoungPeopleEmpathy.ranking(ada.feature_importances_, colnames);


        model = LogisticRegression()
        # create the RFE model and select 3 attributes
        rfe = RFE(model, 3)
        rfe = rfe.fit(train_data_mix, train_Y)
        #column names sorted by ranking
        ranks["RFE"] = YoungPeopleEmpathy.ranking(list(map(float, rfe.ranking_)), colnames, order=-1)


        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
        rf.fit(train_data_mix,train_Y)
        ranks["RF"] = YoungPeopleEmpathy.ranking(rf.feature_importances_, colnames);


        r = {}
        for name in colnames:
            r[name] = round(np.mean([ranks[method][name] 
                                     for method in ranks.keys()]), 2)

        methods = sorted(ranks.keys())
        ranks["Mean"] = r
        methods.append("Mean")

        #print("\t%s" % "\t".join(methods))
        #for name in colnames:
         #   print("%s\t%s" % (name, "\t".join(map(str, 
         #                        [ranks[method][name] for method in methods]))))
        # Put the mean scores into a Pandas dataframe
        meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

        # Sort the dataframe
        meanplot = meanplot.sort_values('Mean Ranking', ascending=False)


        return meanplot
    
    
    
    def tuneParameters(train_data_mix_fin_n,train_Y):
        parameter_candidates = [
    {'C': [1.566,1.691,1.7,1.9,3,3.56,5], 'gamma': [0.00916,0.00888,0.009,0.00999,0.009785,0.01,0.015,0.05],'kernel': ['rbf']},
        ]

        # Create a classifier object with the classifier and parameter candidates
        clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates,cv=10)

        # Train the classifier on data1's feature and target data
        clf.fit(train_data_mix_fin_n, train_Y)
        # View the accuracy score
        print('Best score for data:', clf.best_score_) 

        # View the best parameters for the model found using grid search
        print('Best C:',clf.best_estimator_.C) 
        best_C = clf.best_estimator_.C
        print('Best Kernel:',clf.best_estimator_.kernel)
        best_kernel = clf.best_estimator_.kernel
        print('Best Gamma:',clf.best_estimator_.gamma)
        best_gamma = clf.best_estimator_.gamma
        print('Best Degree:',clf.best_estimator_.degree)
        best_degree = clf.best_estimator_.degree
        return best_C,best_kernel,best_gamma,best_degree
    
    def runXGBoost(train_data_mix_n,train_Y,test_data_mix_n,test_Y):
        xgb = XGBClassifier(max_depth = 10, min_child_weight = 6,gamma = 0.5,colsample_bytree = 0.7, subsample =             0.7,reg_alpha = 1)

        xgb.fit(train_data_mix_n,train_Y)

        predicted_label = xgb.predict(test_data_mix_n)
        print("Test accuracy using XGBoost")
        print(accuracy_score(test_Y, predicted_label))
    
    
    def preprocessAndPredictEmpathy():
        train_data = YoungPeopleEmpathy.loadData()
        YoungPeopleEmpathy.run_baseline_classifier(train_data)
        #Split data
        train_data_reduced = train_data.dropna(subset=set(['Empathy']))
        train_X, test_X, train_Y, test_Y,num_cols,cat_cols =        YoungPeopleEmpathy.splitDataTrainTest(train_data_reduced)
        train_X_numeric,test_X_numeric = YoungPeopleEmpathy.fillNumericMissingValues(train_X,test_X)
        train_X_cat_encoded,test_X_cat_encoded = YoungPeopleEmpathy.encodeCategoricalData(train_X,test_X,cat_cols)
        train_X_numeric,test_X_numeric = YoungPeopleEmpathy.calculateBMI(train_X_numeric,test_X_numeric)
        #joining numeric and categorical 

        train_data_mix = train_X_numeric.join(train_X_cat_encoded)
        test_data_mix = test_X_numeric.join(test_X_cat_encoded)
        # calculate Rank based features
        colnames = train_data_mix.columns
        #0.3
        threshold=0.35

        meanplot = YoungPeopleEmpathy.calculateRankMatrix(train_data_mix,train_Y,colnames,threshold)
        column_name = []
        low_columns = []
        for index, row in meanplot.iterrows():
            if row['Mean Ranking'] >= threshold:
                column_name.append(row['Feature'])
            if row['Mean Ranking'] <= 0.2:
                low_columns.append(row['Feature'])
        top_columns = column_name
        #filter on basis of top-columns
        train_data_mix_n = train_data_mix[top_columns].join(train_data_mix[low_columns])
        test_data_mix_n = test_data_mix[top_columns].join(test_data_mix[low_columns])
        
        """#convert selected features to range data
        num_cols,cat_cols = YoungPeopleEmpathy.getListofNumericColumnsAndCategorical(train_data_mix_n)
        df_encoded_train,df_encoded_test = YoungPeopleEmpathy.getRangeData(train_data_mix_n,test_data_mix_n,num_cols)
        
        #join range data
        train_data_mix_n_red = train_data_mix_n.drop(num_cols,axis=1)
        test_data_mix_n_red = test_data_mix_n.drop(num_cols,axis=1)


        train_data_mix_fin = train_data_mix_n_red.join(df_encoded_train)
        test_data_mix_fin = test_data_mix_n_red.join(df_encoded_test)
        
        #get new top columns
        colnames = train_data_mix_fin.columns
        #0.34
        threshold=0.34
        meanplot = YoungPeopleEmpathy.calculateRankMatrix(train_data_mix_fin,train_Y,colnames,threshold)
        column_name = []
        for index, row in meanplot.iterrows():
            if row['Mean Ranking'] >= threshold:
                column_name.append(row['Feature'])
        top_columns = column_name
        
        train_data_mix_fin_n = train_data_mix_fin[top_columns]
        test_data_mix_fin_n = test_data_mix_fin[top_columns]"""
        
        
        
        
        
        #cross validation using grid search for hyper-parameter tuning
        best_C,best_kernel,best_gamma,best_degree = YoungPeopleEmpathy.tuneParameters(train_data_mix_n,train_Y)
        #using tuned parameters to fit the entire training set
        sv = svm.SVC(kernel=best_kernel,C=best_C,gamma=best_gamma)
        sv.fit(train_data_mix_n,train_Y)
        #predicting on test data
        predicted_label = sv.predict(test_data_mix_n)
        print("Test Accuracy using SVM")
        print(accuracy_score(test_Y, predicted_label))
        YoungPeopleEmpathy.runXGBoost(train_data_mix_n,train_Y,test_data_mix_n,test_Y)