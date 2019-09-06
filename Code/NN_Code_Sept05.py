# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:45:49 2019

@author: kenne
"""
import pandas as pd
import random
import torch

# The .csv file contains all of the data used in the neural network
# The last 3 rows in the file contain the one-hot vectors which indicate the
# optimal portfolio allocation.
NNdata = pd.read_csv('NNdata.csv', index_col=0)

# Randomly select training and test sets
def datasep(df):
    """
    Take a pandas dataframe, ready for a neural network (i.e. last  3 rows are
    the one-hot vector labels), and randomly select a test set. The remaining 
    data willbe the training set.
    Currently the separation is 80:20 for the training set, but this is easily
    changed.
    """
    k = len(df.T)
    cols = [i for i in range(k)]
    random.shuffle(cols)
    j = round(k/5)
    test = cols[:j]
    test.sort()
    testdf = df.T.iloc[test]
    testdf = testdf.T
    train = cols[j:]
    train.sort()
    traindf = df.T.iloc[train]
    traindf = traindf.T
    return [testdf, traindf]

dfs = datasep(NNdata)
testset = torch.tensor(dfs[0].T.values)
trainset = torch.tensor(dfs[1].T.values)




# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, D_out = len(trainset), len(NNdata)-3, 500, 50, 3

# Create Tensors to hold inputs and outputs
x = trainset.narrow(1, 0, D_in).type(torch.FloatTensor)
y = trainset.narrow(1, D_in, 3).type(torch.FloatTensor)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),    # To add more layers, we add more Linears and
    torch.nn.ReLU(),            # also more ReLU's
    torch.nn.Linear(H2, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum') # Try testing other loss functions

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4 # This is the learning rate used in the gradient descent
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())   #This is here to check that the code works

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    
    
# I haven't yet coded the testing of the model. I will do this soon.