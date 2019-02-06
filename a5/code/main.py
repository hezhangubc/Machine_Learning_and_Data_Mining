import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2
from pca import PCA, AlternativePCA, RobustPCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")
        
        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=1)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)
        
        # kernel logistic regression with a poly kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_poly, lammy=0.01, p=2)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegPolyKernel.png")
        
        # kernal logistic regression with a RBF kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=0.01, sigma=0.5)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))
        
        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel.png")
        

    elif question == "1.2":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # hyperparameter choose 
        m1 = np.array([-2,-1,0,1,2])
        m2 = np.array([-4,-3,-2,-1,0])
        
        ite1 = m1.shape
        ite2 = m2.shape
        
        sigma_record = np.zeros(ite1)
        lammy_record = np.zeros(ite2)
        
        training_error = np.zeros(25)
        validation_error = np.zeros(25)
        
        sigma_argmin_train = 0 
        lammy_argmin_train = 0
        
        sigma_argmin_validation = 0
        lammy_argmin_validation = 0
        
        min_train = 1000000
        min_validation = 100000
        
        index = 0
        for i in range(5):
            sigma_record[i] = 10**float(m1[i])
            
            for j in range(5):
                lammy_record[j] = 10**float(m2[j])
                
                lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=lammy_record[j], sigma=sigma_record[i])
                lr_kernel.fit(Xtrain, ytrain)
                
                training_error[index] = np.mean(lr_kernel.predict(Xtrain) != ytrain)
                validation_error[index] = np.mean(lr_kernel.predict(Xtest) != ytest)
                
                if training_error[index] < min_train:
                    min_train = training_error[index]
                    sigma_argmin_train = sigma_record[i]
                    lammy_argmin_train = lammy_record[j]
                if validation_error[index] < min_validation:
                    min_validation = validation_error[index]
                    sigma_argmin_validation = sigma_record[i]
                    lammy_argmin_validation = lammy_record[j]
                index += 1
                
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=lammy_argmin_train, sigma=sigma_argmin_train)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel_min_training.png")
        
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=lammy_argmin_validation, sigma=sigma_argmin_validation)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel_min_validation1.png")
                
        

    elif question == '4.1': 
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5           # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.jpg'.format(i))

    else:
        print("Unknown question: %s" % question)    