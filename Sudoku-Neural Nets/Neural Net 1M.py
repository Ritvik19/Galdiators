import numpy as np
import pandas as pd

from keras import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical

import smtplib, ssl

smtp_server = "smtp.gmail.com"
port = 587
with open('E:/API-Credentials/Gmail.txt') as f:
    data = list(map(lambda x: x.strip(), f.read()))
    sender_email = data[0]
    receiver_email = data[0]
    password = data[1]
context = ssl.create_default_context()
try:
    server = smtplib.SMTP(smtp_server,port)
    server.starttls(context=context) # Secure the connection
    server.login(sender_email, password)
except Exception as e:
    print(e)


def load_data(n=1000000):
    """
    Function to load data in the keras way.
    
    Parameters
    ----------
    n (int): Number of  examples
    
    Returns
    -------
    Xtrain, ytrain (np.array, np.array),
        shapes (0.8*n, 9, 9), (0.8*n, 9, 9): Training samples
    Xtest, ytest (np.array, np.array), 
        shapes (0.2*n, 9, 9), (0.2*n, 9, 9): Testing samples
    """
    sudokus = pd.read_csv('../../[Data] Sudoku/sudoku.csv').sample(n).reset_index(drop=True).values
        
    quizzes, solutions = sudokus.T
    flatX = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in quizzes])
    flaty = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in solutions])
    threshold = int(0.8*n)
    return (flatX[:threshold], flaty[:threshold]), (flatX[threshold:], flaty[threshold:])

def diff(grids_true, grids_pred):
    """
    This function shows how well predicted quizzes fit to actual solutions.
    It will store sum of differences for each pair (solution, guess)
    
    Parameters
    ----------
    grids_true (np.array), shape (?, 9, 9): Real solutions to guess in the digit repesentation
    grids_pred (np.array), shape (?, 9, 9): Guesses
    
    Returns
    -------
    diff (np.array), shape (?,): Number of differences for each pair (solution, guess)
    """
    return (grids_true != grids_pred).sum((1, 2))

def delete_digits(X, n_delet=1):
    """
    This function is used to create sudoku quizzes from solutions
    
    Parameters
    ----------
    X (np.array), shape (?, 9, 9, 9|10): input solutions grids.
    n_delet (integer): max number of digit to suppress from original solutions
    
    Returns
    -------
    grids: np.array of grids to guess in one-hot way. Shape: (?, 9, 9, 10)
    """
    grids = X.argmax(3)  # get the grid in a (9, 9) integer shape
    for grid in grids:
        grid.flat[np.random.randint(0, 81, n_delet)] = 0  # generate blanks (replace = True)
        
    return to_categorical(grids)

def batch_smart_solve(grids, solver):
    """   
    This function solves quizzes in the "smart" way. 
    It will fill blanks one after the other. Each time a digit is filled, 
    the new grid will be fed again to the solver to predict the next digit. 
    Again and again, until there is no more blank
    
    Parameters
    ----------
    grids (np.array), shape (?, 9, 9): Batch of quizzes to solve (smartly ;))
    solver (keras.model): The neural net solver
    
    Returns
    -------
    grids (np.array), shape (?, 9, 9): Smartly solved quizzes.
    """
    grids = grids.copy()
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))  # get predictions
        probs = preds.max(2).T  # get highest probability for each 81 digit to predict
        values = preds.argmax(2).T + 1  # get corresponding values
        zeros = (grids == 0).reshape((grids.shape[0], 81))  # get blank positions

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):  # don't try to fill already completed grid
                where = np.where(zero)[0]  # focus on blanks only
                confidence_position = where[prob[zero].argmax()]  # best score FOR A ZERO VALUE (confident blank)
                confidence_value = value[confidence_position]  # get corresponding value
                grid.flat[confidence_position] = confidence_value  # fill digit inplace
    return grids

input_shape = (9, 9, 10)
(_, ytrain), (Xtest, ytest) = load_data()  # We won't use _. We will work directly with ytrain
print('Data Loaded')
# one-hot-encoding --> shapes become :
# (?, 9, 9, 10) for Xs
# (?, 9, 9, 9) for ys
Xtrain = to_categorical(ytrain).astype('int32')  # from ytrain cause we will creates quizzes from solusions
Xtest = to_categorical(Xtest).astype('int32')

ytrain = to_categorical(ytrain-1).astype('int32') # (y - 1) because we 
ytest = to_categorical(ytest-1).astype('int32')   # don't want to predict zeros

print('Data Transformed')

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())

grid = Input(shape=input_shape)  # inputs
features = model(grid)  # commons features

# define one Dense layer for each of the digit we want to predict
digit_placeholders = [
    Dense(9, activation='softmax')(features)
    for i in range(81)
]

solver = Model(grid, digit_placeholders)  # build the whole model
solver.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print('Model Compiled')
ckpt = ModelCheckpoint('Sudoku1M.h5', monitor='loss', verbose=1, save_best_only=False, mode='auto')
model.summary()

solver.fit(
    delete_digits(Xtrain, 0),  # we don't delete any digit for now
    [ytrain[:, i, j, :] for i in range(9) for j in range(9)],  # each digit of solution
    batch_size=128,
    verbose=0,
    epochs=1,  # 1 epoch should be enough for the task
    callbacks=[ckpt]
)
try:
    server.sendmail(sender_email, receiver_email, 'Model Trained: 0')
except Exception as e:
    print(e)

early_stop = EarlyStopping(patience=2, verbose=1)

i = 1
for nb_epochs, nb_delete in zip(
        [2, 5, 5, 5, 7, 10, 15, 15, 15, 15, 15, 15, 15, 15, 15, 20, 20, 20, 20],  # epochs for each round
        [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]  # digit to pull off
):
    print(f'Pass {i}: {nb_delete}')
    i += 1
    
    solver.fit(
        delete_digits(Xtrain, nb_delete),  # delete digits from training sample
        [ytrain[:, i, j, :] for i in range(9) for j in range(9)],
        validation_data=(
            delete_digits(Xtrain, nb_delete), # delete same amount of digit from validation sample
            [ytrain[:, i, j, :] for i in range(9) for j in range(9)]),
        batch_size=128,
        epochs=nb_epochs,
        verbose=0,
        callbacks=[early_stop, ckpt]
    )
    try:
        server.sendmail(sender_email, receiver_email, 'Model Trained: {nb_delete}')
    except Exception as e:
        print(e)

quizzes = Xtest.argmax(3)  # quizzes in the (?, 9, 9) shape. From the test set
true_grids = ytest.argmax(3) + 1  # true solutions dont forget to add 1 
smart_guesses = batch_smart_solve(quizzes, solver)  # make smart guesses !

deltas = diff(true_grids, smart_guesses)  # get number of errors on each quizz
accuracy = (deltas == 0).mean()  # portion of correct solved quizzes

print(
"""
Grid solved:\t {}
Correct ones:\t {}
Accuracy:\t {}
""".format(
deltas.shape[0], (deltas==0).sum(), accuracy
)
)

solver.save('Sudoku1M.h5')

finally:
    server.quit() 
