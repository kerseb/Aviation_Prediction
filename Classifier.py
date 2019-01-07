from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score

import os
import numpy
import csv



class NeuralNetwork:
    'A simple feed-forward neural network class based on keras with tensorflow backend'
    model = Sequential()
    #input_dim = 0
    #output_dim = 0
    #layer = 0
    #activation_fun = []   
    #neurons = []
    #loss_fun = ''
    #optimizer = ''
    #metrics = ''
    
    def __init__(self, in_dim = 1, out_dim = 1, hidden_layer = 1, neur = 1, 
                 act = 'relu',loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'], rand_seed = False, load_path = ''):
        
        if(load_path != ''):
            self.load_network(load_path)
            return
        
        if rand_seed:
            numpy.random.seed(7)
        
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.layer = hidden_layer +2
        self.loss_fun = loss_fun
        self.optimizer = optimizer 
        self.metrics = metrics
        
        if isinstance(act, list):
            self.activation_fun = act
        else:
            self.activation_fun = []
            for i in range(self.layer-1):
                self.activation_fun.append(act)
            self.activation_fun.append('sigmoid')
            
        if isinstance(neur, list):
            self.neurons = neur
        else:
            self.neurons = []
            for i in range(self.layer-1):
                self.neurons.append(neur)
        
        self.add_layer()
        self.model.compile(loss=self.loss_fun, optimizer=self.optimizer, metrics=self.metrics)
        
    def add_layer(self):        
        #initialize input layer
        self.model.add(Dense(self.neurons[0], input_dim=self.input_dim, activation=self.activation_fun[0]))
        #initialize hidden layer
        for i in range(1,self.layer -1):
            self.model.add(Dense(self.neurons[i], activation=self.activation_fun[i]))
        #initialize output layer
        if self.output_dim > 1:
            self.model.add(Dense(self.output_dim, activation="softmax"))
        else:
            self.model.add(Dense(self.output_dim, activation=self.activation_fun[-1]))
        return True
        
    def fit(self,X,Y,epochs = 300, batch_size = 32):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size)
        return True

    
    def evaluate(self,X,Y):
        """
        if self.output_dim > 1:
            print(numpy.size(X,1))
            scores = 0;
            for i in range(numpy.size(X,1)):
                inp_val = numpy.resize(X[i,:],(1,numpy.size(X[i,:],0)))
                am = self.predict(inp_val)
                scores += Y[i,am]
            scores /= numpy.size(X,1)
        else:
        """
        scores = self.model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        return scores
    
    def predict(self,X):
        if self.output_dim > 1:
            Y = self.model.predict(X)
            return numpy.argmax(Y) , Y
        else:
            return self.model.predict(X)
    
    def save_network(self,path = ''):
        abs_path = os.path.abspath(path)

        try:
            os.mkdir(abs_path)
        except FileExistsError:
            pass
        
        abs_path += '/'
        
        # save network parameter
        model_par = open(abs_path + 'model_par.csv','w')
        parameter =  csv.writer(model_par)
        parameter.writerow([self.input_dim])
        parameter.writerow([self.output_dim])
        parameter.writerow([self.layer])
        parameter.writerow(self.activation_fun)   
        parameter.writerow(self.neurons)
        parameter.writerow([self.loss_fun])
        parameter.writerow([self.optimizer])
        parameter.writerow(self.metrics)   
        model_par.close()
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(abs_path + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(abs_path + "model.h5")
        print("Saved model to disk")
        return True
    
    def load_network(self,path = ''):
        abs_path = os.path.abspath(path)
        abs_path += '/'
        
        # load model parameter 
        if os.path.exists(abs_path + 'model_par.csv'): 
            model_par = open(abs_path + 'model_par.csv','r')
            parameter =  csv.reader(model_par, delimiter='\n')
            self.input_dim = parameter.__next__()[0]
            self.output_dim = parameter.__next__()[0]
            self.layer = parameter.__next__()[0]
            self.activation_fun = parameter.__next__()   
            self.neurons = parameter.__next__()
            self.loss_fun = parameter.__next__()
            self.optimizer = parameter.__next__()[0]
            self.metrics = parameter.__next__()
            model_par.close()      
        else:
            print('File :' + abs_path + 'model_par.csv' + ' does not exists')
            return False
        
        
        
        # load json and create model
        if os.path.exists(abs_path + 'model.json'): 
            json_file = open(abs_path + 'model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        else:
            print('File :' + abs_path + 'model.json' + ' does not exists')
            return False
        # load weights into new model
        if os.path.exists(abs_path + 'model.h5'): 
            self.model.load_weights(abs_path + 'model.h5')
        else:
            print('File :' + abs_path + 'model.h5' + ' does not exists')
            return False
        
        # compile model
        self.model.compile(loss=self.loss_fun, optimizer=self.optimizer, metrics=self.metrics)
        
        print("Loaded model from disk")
        return True

    def get_model_param(self):
        return {'input dimension':self.input_dim,
                'output dimension':self.output_dim,
                'hidden layer':self.layer,
                'activation functions':self.activation_fun,
                'number of neurons':self.neurons,
                'loss function':self.loss_fun,
                'optimizer':self.optimizer,
                'metrics':self.metrics}


    
    
    
def cross_validation_NN(X,Y,in_dim, out_dim,hidden_layer,neurons,epochs = 300,batch_size = 32, activation_fun = 'relu', loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'],rand_seed = True,onehot_encoder = 0):
    # make hidden_layer and neurons iterable if they are just a number
    if not isinstance(hidden_layer, list):
        hidden_layer = [hidden_layer]
    if not isinstance(neurons, list):
        neurons = [neurons]
    
    """ makes problems in case y is multidimensional
    # get input and output dimension
    try:
        input_dim = numpy.size(X,1)    
    except IndexError:
        input_dim = 1
    try:
        output_dim = numpy.size(Y,1)    
    except IndexError:
        output_dim = 1
    """


    # initiate cross-validation scores and parameter list, and get cv indices
    cv_scores = []
    param = []
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(X, Y))

    # iterate over each parameter combination
    for h in hidden_layer:
        for n in neurons:
            
            #initilize NN
            NN = NeuralNetwork(in_dim = in_dim, out_dim = out_dim, hidden_layer = h, neur = n, act = activation_fun,loss_fun = loss_fun, optimizer = optimizer , metrics = metrics, rand_seed = rand_seed)
            
            print(NN.get_model_param())

            scores = []
            """
            scores = cross_val_score(NN, X, Y, cv=5, scoring='neg_mean_squared_error')
            """

            for j, (train_idx, val_idx) in enumerate(folds):
                print('\nFold ',j+1)
                # splitt data
                X_train_cv = X[train_idx]
                Y_train_cv = Y[train_idx]
                X_valid_cv = X[val_idx]
                Y_valid_cv = Y[val_idx]

                if onehot_encoder != 0:
                    Y_train_cv = onehot_encoder.transform(Y_train_cv)
                    Y_valid_cv = onehot_encoder.transform(Y_valid_cv)
            
                # train NN
                NN.fit(X_train_cv,Y_train_cv,epochs = epochs, batch_size = batch_size)
                # evaluate NN
                print(NN.evaluate(X_valid_cv,Y_valid_cv)[0])
                scores.append(NN.evaluate(X_valid_cv,Y_valid_cv)[0])
            cv_scores.append(numpy.array(scores).mean())
            param.append([h,n])

            del NN

    optimal_p = param[cv_scores.index(min(cv_scores))]
    optimal_h = optimal_p[0]
    optimal_n = optimal_p[1]
        
    print("Optimal number of hidden layer is: %i" % (optimal_h))
    print("Optimal number of neurons is: %s" % (optimal_n))
    return optimal_h , optimal_n, min(cv_scores) 
    
    
    
    
    
