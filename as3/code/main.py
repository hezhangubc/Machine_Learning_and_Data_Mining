
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # MY CODE HERE FOR Q1.1.1
        star_record = np.sum(X, axis = 0)   #sum all the stars of each item
        item_num_with_max_star = star_record.argmax()
        item_with_max_star = item_inverse_mapper[item_num_with_max_star]
        print("the item with the most total stars:", item_with_max_star)
        print("the total stars of the item:",star_record[0,item_num_with_max_star])
        
        # YOUR CODE HERE FOR Q1.1.2
        user_record = X.getnnz(axis = 1)
        user_most_item = user_record.argmax()
        number_of_items = user_record.max()
        user_id_most_item = user_inverse_mapper[user_most_item]
        print("the user that has rated the most items:", user_id_most_item)
        print("the number of items the user rated:", number_of_items)
        
        # YOUR CODE HERE FOR Q1.1.3
        # The number of ratings per user
        plt.title("The number of ratings per user")
        plt.hist(user_record)
        plt.yscale('log', nonposy='clip')
        
        fname = os.path.join("..", "figs", "q1_1_2_hist_numberofratings_per_user.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # the number of ratings per item 
        item_record = X.getnnz(axis = 0)
        plt.title("The number of ratings per item")
        plt.hist(item_record, bins = 5)
        plt.yscale('log', nonposy='clip')
        
        fname = os.path.join("..", "figs", "q1_1_2_hist_numberofratings_per_item.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # the rating itself
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            rating = pd.read_csv(f,usecols=[2])
            rating = rating.values
        plt.title("The ratings themselves")
        plt.hist(rating)
        fname = os.path.join("..", "figs", "q1_1_2_hist_ratings_themselves.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)
        X = X.T   # transform X from a matrix form to a nparray form
        
        
        #### YOUR CODE HERE FOR Q1.2
        # euclidean based neareast neighbor
        nbrs = NearestNeighbors(n_neighbors = 6, algorithm = 'auto', metric = 'euclidean').fit(X)
        nearest_index_1 = nbrs.kneighbors(grill_brush_vec.T, 6, return_distance = False)
        print("The five nearest neighbors of grill brush:", nearest_index_1[0,1:])
        
        # normalized euclidean distance 
        X_normalized = normalize(X, axis = 1)
        nbrs = NearestNeighbors(n_neighbors = 6, algorithm = 'auto', metric = 'euclidean').fit(X_normalized)
        nearest_index_2 = nbrs.kneighbors(grill_brush_vec.T, 6, return_distance = False)
        print("The five nearest neighbors of grill brush:", nearest_index_2[0,1:])
        # cosine similarity
        nbrs = NearestNeighbors(n_neighbors = 6, algorithm = 'auto', metric = 'cosine').fit(X)
        nearest_index_3 = nbrs.kneighbors(grill_brush_vec.T, 6, return_distance = False)
        print("The five nearest neighbors of grill brush:", nearest_index_3[0,1:])
        
        
        #### YOUR CODE HERE FOR Q1.3
        item_record = X.getnnz(axis = 0)
        print("Number of reviews for the 5 recommendations based on euclidean distance:", item_record[nearest_index_1[0,1:]])
        print("Number of reviews for the 5 recommendations based on cosine similarity:", item_record[nearest_index_3[0,1:]])
    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)
        
        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # YOUR CODE HERE
        z = np.zeros((500))
        z[0:100] = 1
        z[100:500] = 0.1
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,z)
        print(model.w)
        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="Weighted_least_squares_outliers.pdf")
        
    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']
 
        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)
        
        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="least_squares_with_bias.pdf")
    
    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares with Basis",filename="least_squares_with_poly_basis.pdf")
            
            

    else:
        print("Unknown question: %s" % question)

