
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'cleaned_train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'cleaned_test.csv' #replace
ALPHA = 1e-3
EPOCHS = 5000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model2'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    col=X.shape
    arr=np.ones((col[0],1))

    nx=np.hstack((arr,X))
    print nx.shape
    return nx
    #pass#remove this line once you finish writing




 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    arr=np.zeros(n_thetas)
    return arr
  #  pass



def train(theta, X, y, model):
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #theta=theta.reshape((theta.shape[0],1))
     #print theta.shape
     for i in range(EPOCHS):
        p_y=predict(X,theta)
        #print(p_y)
        cost=costFunc(m,y,p_y)
        J.append(cost)
        cal_gra=calcGradients(X,y.flatten(),p_y.flatten(),m)

        
        #print cal_gra.shape
        theta=makeGradientUpdate(theta,cal_gra)
             
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)

     model['J'] = J
     model['theta'] = list(theta)
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    #pass
    k=y_predicted-y
    k=np.multiply(k,k)
    #k=k*k
    su=np.sum(k)
    m=2*m
    n=su/m
    return n

def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
    # pass
    #y_predicted = y_predicted.flatten()
    d1=y_predicted-y
    print y_predicted.shape
    print d1.shape
    d1=d1.reshape((d1.shape[0],1))
    d2=X*d1
    return np.sum(d2,axis=0)/m

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    a=ALPHA*grads
    
    return theta-a


#this function will take two paramets as the input
def predict(X,theta):
    return np.dot(X,theta)
    #pass


########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))

    else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            X = appendIntercept(X)
            accuracy(X,y,model)

if __name__ == '__main__':
    main()
