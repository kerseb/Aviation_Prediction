import csv
import numpy as np
import Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


train_path = "data/train.csv"
test_path = "data/test.csv"
output_path = "data/submission.csv"




def import_csv_to_list(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        #i = 1
        for row in reader:
            data.append(row)
            #i += 1
            #if i > 1000000:
            #    break
    return data

def train_NN(data):
    data =  np.array(data)
    X = data[1:,2:-1]
    Y = data[1:,-1]

    # integer encoding
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    #print(Y)
    Y = Y.reshape(len(Y), 1)

    #onehot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(Y)
    
    """
    print(np.size(Y,1))
    print(Y[1])
    NN = Classifier.NeuralNetwork(in_dim = np.size(X,1), out_dim = np.size(Y,1), hidden_layer = 2, neur = 5, act = 'relu',loss_fun = 'mean_squared_error', 
        optimizer = 'adam' , metrics = ['accuracy'], rand_seed = False)
    
    NN.fit(X,Y,epochs = 1, batch_size = 32)

    tv = X[1,:]
    #print(np.size(tv,0))
    tv = np.resize(tv,(1,np.size(tv,0)))
    #print(tv)
    a1, a2 = NN.predict(tv)
    print(a1)
    print(a2)
    

    v = NN.evaluate(tv,np.resize(Y[1],(1,4)))
    print("v : == ")
    print(v)

    #NN.evaluate(X,Y)
    """
    
    optimal_h , optimal_n, cv_scores  = Classifier.cross_validation_NN(X,Y,in_dim = np.size(X,1), out_dim = 4,hidden_layer = [5,15,30],neurons = [5,25,50],epochs = 30,batch_size = 32, activation_fun = 'relu', loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'],rand_seed = True,onehot_encoder = onehot_encoder)
    print("---------------Values of CV: --------------")
    print(optimal_h)
    print(optimal_n)
    print(cv_scores)
    print("-------------------------------------------")
    NN = Classifier.NeuralNetwork(in_dim = np.size(X,1), out_dim = np.size(Y,1), hidden_layer = optimal_h, neur = optimal_n, act = 'relu',loss_fun = 'mean_squared_error', 
        optimizer = 'adam' , metrics = ['accuracy'], rand_seed = False)
    
    Y = onehot_encoder.transform(Y)
    NN.fit(X,Y,epochs = 1, batch_size = 32)

 
    tv = X[1,:]
    #print(np.size(tv,0))
    tv = np.resize(tv,(1,25))
    #print(tv)
    a1, a2 = NN.predict(tv)
    print(a1)
    print(a2)
    

    return NN

def main():
    train_data = import_csv_to_list(train_path)
    NN_classifier = train_NN(train_data)


if __name__ == '__main__':
    main()

