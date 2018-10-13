# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier #for plotting 3.3 with scikit-learn 


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        
        train_errors = np.zeros(14)
        test_errors = np.zeros(14)

        depths = np.arange(1,15) # depths to try
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            
            print("Training error: %.3f" % tr_error)
            print("Testing error: %.3f" % te_error)
        
            train_errors[i] = tr_error
            test_errors[i] = te_error

        x_vals = np.arange(1, 15)
        plt.title("The effect of tree depth on testing/training error")
        plt.plot(x_vals, train_errors, label="training error")
        plt.plot(x_vals, test_errors, label="testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = os.path.join("..", "figs", "q1_1_TrainTest.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape
        
        
        #Xvalidation = X[:n//2]
        #yvalidation = y[:n//2]
        #Xtraining = X[n//2:]
        #ytraining= y[n//2:]
        
        Xtraining= X[:n//2]
        ytraining = y[:n//2]
        Xvalidation = X[n//2:]
        yvalidation= y[n//2:]
        
        validation_errors = np.zeros(14)

        depths = np.arange(1,15) # depths to try
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(Xtraining, ytraining)

            y_pred = model.predict(Xvalidation)
            val_error = np.mean(y_pred != yvalidation)
            
            print("Testing error: %.3f" % te_error)

            validation_errors[i] = val_error

        x_vals = np.arange(1, 15)
        plt.title("The effect of tree depth on testing/training error")
        plt.plot(depths, validation_errors, label="validation error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = os.path.join("..", "figs", "q1_2_validation.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        print("Best depth:", depths[np.argmin(validation_errors)])     


    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        # mycode
        # 2.2.1 
        X[50]
        # 2.2.2
        wordlist[X[500]==1]
        # 2.2.3 
        groupnames[y[500]]
       


    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (our code) validation error: %.3f" % v_error)


        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (sklearn) validation error: %.3f" % v_error)
    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
        
        #k=1, training error
        model = KNN(k=1)
        model.fit(X, y)
        y_pred = model.predict(X)
        v_error = np.mean(y_pred != y)
        print("KNN (k=1) training error: %.3f" % v_error)
        
        #k=1, test error
        model = KNN(k=1)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        v_error = np.mean(y_pred != ytest)
        print("KNN (k=1) test error: %.3f" % v_error)


        #k=3, training error
        model = KNN(k=3)
        model.fit(X, y)
        y_pred = model.predict(X)
        v_error = np.mean(y_pred != y)
        print("KNN (k=3) training error: %.3f" % v_error)
        
        #k=3, test error
        model = KNN(k=3)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        v_error = np.mean(y_pred != ytest)
        print("KNN (k=3) test error: %.3f" % v_error)

        #k=10, training error
        model = KNN(k=10)
        model.fit(X, y)
        y_pred = model.predict(X)
        v_error = np.mean(y_pred != y)
        print("KNN (k=10) training error: %.3f" % v_error)
        
        #k=10, test error
        model = KNN(k=10)
        model.fit(X, y)
        y_pred = model.predict(Xtest)
        v_error = np.mean(y_pred != ytest)
        print("KNN (k=10) test error: %.3f" % v_error)

        
        # 3.3 plot with my implementation of KNN and k=1
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q3_3_myKNN(k=1).pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # 3.3 plot withscikit-learn KNneighbors and k=1
        model= KNeighborsClassifier(n_neighbors=1)
        model.fit(X, y) 
        y_pred = model.predict(X)
        t_error = np.mean(y_pred != y)
        print("KNeighbors (k=1) training error: %.3f" % v_error)
        utils.plotClassifier(model, X, y)
        fname2 = os.path.join("..", "figs", "q3_3_KNeighbors(k=1).pdf")
        plt.savefig(fname2)
        print("\nFigure saved as '%s'" % fname)
        
        


    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        
        
        # TODO: code here
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))

        print("sklearn implementations")
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info gain, more trees")
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))




    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        
        # for the first question: print the error of the final clustering model
        model = Kmeans(k=4)
        model.fit(X)
        dist_error_final = model.error(X)
        print (dist_error_final)
        
        # for the third question: print the figure with the lowest error
        model_best = None
        error_minimum = np.inf
        for i in range(50):
            model = Kmeans(4)
            model.fit(X)
            error = model.error(X)
            if error <= error_minimum:
                error_minimum = error
                model_best = model
        y = model_best.predict(X)
        plt.scatter(X[:,0   ], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_lowest_error.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
                        
            
            

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        ''' for the third question: hand in a plot of the minimum error found across 50 random initializations, 
            as a function of k, taking k from 1 to 10
        '''
        error_record = np.zeros(10)
        for k in range(10):
            model_best = None
            error_minimum = np.inf
            for i in range(50):
                model = Kmeans(k+1)
                model.fit(X)
                error = model.error(X)
                if error < error_minimum:
                    error_minimum = error
                    model_best = model
            error_record[k] = error_minimum
        plt.plot(np.array(range(1,11)), error_record)
        plt.xlabel('k')
        plt.ylabel("Error")
        fname = os.path.join("..", "figs", "error_with_k.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
                



    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=3, min_samples=2)
        y = model.fit_predict(X)
        # 

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
