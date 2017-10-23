import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
    
def load_log_file(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        result = [[line[0],float(line[3])] for line in reader]
        return result
    
def load_log_files():
    folders = [0,1,3,4,4,4,4,4,4,4,4,5]
    result = []
    
    for ii in folders:
        file = './data/{0}/driving_log.csv'.format(ii)
        result += load_log_file(file)    
    return result
    
logs = load_log_files()
    
X_train = np.array([cv2.imread(log[0]) for log in logs])
y_train = np.array([log[1] for log in logs])

X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.1)

batch_size = 128
print(len(y_train) // batch_size)

def generator(X,y):
    while True:
        X,y = shuffle(X,y) 
        for i in range(0,len(y) // batch_size):
            inputs = X[i * batch_size : (i+1) * batch_size]
            targets = y[i * batch_size : (i+1) * batch_size]
            yield (inputs,targets)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(3 * 6, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(3 * 16, (5, 5), activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2 * 120,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2 * 84,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    generator(X_train,y_train),
    len(y_train) // batch_size,
    epochs=10,
    validation_data = (X_valid,y_valid))

#model.fit(X_train,y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')

    

        

