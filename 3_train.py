import numpy as np
import cv2
from model import *
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split


def train(feature, label):

    model = MLPEncoder_NOXY()

    # move model to gpu
    model.to('cuda:0')

    # initial optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
    loss_fn = torch.nn.MSELoss()

    # initial dataset
    batch_size = 32
    tactile_dataset = TensorDataset(feature, label)

    m = len(tactile_dataset)

    train_data, val_data = random_split(tactile_dataset, [int(m * 0.8), m - int(m * 0.8)])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    # start training
    num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):

        train_loss = train_epoch(model,train_loader,optimizer,loss_fn)
        test_loss = test_epoch(model,valid_loader,loss_fn)

        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, test_loss))

        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(test_loss)

    return model, diz_loss

def train_epoch(model, train_loader, optimizer, loss_fn):

    # Set train mode for the encoder
    model.train()

    total_loss =0
    total_num = 0
    for features, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(features)  # Forward pass.
        loss = loss_fn(logits, labels)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * labels.shape[0]
    total_num += train_loader.dataset.__len__()

    return total_loss / total_num

def test_epoch(model, train_loader, loss_fn):

    # Set train mode for the encoder
    model.eval()

    total_loss =0
    total_num = 0
    for features, labels in train_loader:
        logits = model(features)  # Forward pass.
        loss = loss_fn(logits, labels)  # Loss computation.
        total_loss += loss.item() * labels.shape[0]
    total_num += train_loader.dataset.__len__()

    return total_loss / total_num

if __name__=="__main__":

    dataset = np.load('data/cal_dataset2.npy',allow_pickle=True)
    dataset = torch.tensor(dataset,dtype=torch.float32,device='cuda:0')
    model, diz_loss = train(dataset[:,:3], dataset[:,5:7])
    torch.save(model,'./model/model_noxy.pt')
    np.save('./model/loss_noxy', diz_loss)