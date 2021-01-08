import sys
sys.path.insert(1, '/home/mattStubbs/github/Tools')
import copy
from ioUtils import CSVData
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

#...................................................................................................#
                                       # TRAINING
#...................................................................................................#

import copy

def train_features(model, iteration_display, log_number, val_number, train_loader, val_loader, num_epochs, learning_rate, dump_path, lr_decay):
    
    #.........................................................................#
    train_log = CSVData(dump_path, "log_train.csv")
    val_log = CSVData(dump_path, "log_val.csv")
    best_val_log = CSVData(dump_path, "log_best_val.csv")
    keys = ['epoch', 'iteration', 'accuracy', 'loss']
    best_model_wts = copy.deepcopy(model.state_dict())

    #.........................................................................#
    epoch = 0.; iteration = 0; correct = 0.; acc = 0.; total = 0.; best_acc = 0.;

    dataloaders = {
        "train": train_loader,
        "validation": val_loader
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    lambd = lambda epoch: lr_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambd)
    
    #.........................................................................#
    for epoch in range(num_epochs):
        # monitor training loss
        train_loss = 0.0
    
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for data, target in train_loader:
            
            model.train()
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            
            #display epoch and iteration at intervals
            if iteration % iteration_display == 0:
                print('epoch: {:.3f}, iteration: {:.2f}'.format(epoch, iteration))

            ##.............................................................##
            output = model.forward(data.float())      # forward pass
            loss = criterion(output, target.long())  # calculate the loss
            loss.backward()                   # propagate loss backwards in network
            optimizer.step()                  # update the gradient weights

            #evaluation parameters
            correct += output.argmax(1).eq(target).sum().item()
            total += target.shape[0]
            acc = correct / total             # running accuracy, loss in training stage

            #.........................................................................#
                                    # update training log
                
            if iteration % log_number == 0 or iteration == len(train_loader)*num_epochs:
                train_log.record(keys, [epoch, iteration, acc, loss])
                train_log.write()

            #.........................................................................#
                                        # VALIDATION
                
            if iteration % val_number == 0 or iteration == len(train_loader)*num_epochs:
                #^ validation interval adjustable settings
                model.eval()
                correctV = 0.; accV = 0.; totalV = 0.;    #reset statistics 
                with torch.no_grad():
                    for dataV, targetV in val_loader:                       #iterate over validation dataset
                        dataV = dataV.to(device)       #send to device
                        targetV = targetV.to(device)
                        outputV = model.forward(dataV.float())        #forward pass
                        lossV = criterion(outputV, targetV.long())   #compute loss

                    #evaluation parameters
                        correctV += outputV.argmax(1).eq(targetV).sum().item()
                        totalV += targetV.shape[0]

                accV = correctV / totalV        #validation accuracy
                
                if accV > best_acc:
                    best_acc = accV
                    print('{}accV: {:.4f}, bestAcc: {:.4f}{}'.format('-'*10, accV, best_acc, '-'*10))
                    best_val_log.record(keys, [epoch, iteration, accV, lossV])
                    best_val_log.write()
                    #save best model parameters
                    torch.save(model.state_dict(), dump_path + '/state_dict')

                val_log.record(keys, [epoch, iteration, accV, lossV])
                val_log.write()
                
        #.........................................................................#
                             # increment iteration, epoch
               
            iteration += 1
            epoch += 1 / len(train_loader)
        
        scheduler.step()
        #print epoch statistics
        print('Epoch: {:.4f} Loss: {:.4f}, accuracy: {:.4f}'.format(epoch, loss, acc))
    
    return model

#...................................................................................................#
#...................................................................................................#
                                        # TESTING
#...................................................................................................#
#...................................................................................................#

def Test(model, test_loader, dump_path):
    model = model
    model.load_state_dict(torch.load(dump_path + '/state_dict'))
    model.eval() # prep model for *evaluation*
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #.........................................................................#
    epoch, iteration =0., 0.
    criterion = nn.CrossEntropyLoss()

    #.........................................................................#
    with torch.no_grad():           

        correctT, totalT, accT, lostT = 0., 0., 0., 0. 
        y_pred, y_true, y_pred_np = [], [], []
        y_pred_unrounded = np.zeros([1,2])
        
        #.........................................................................#
        for dataT, targetT in test_loader:

            dataT, targetT = dataT.to(device), targetT.to(device)
            iteration += 1

            if iteration % 200 == 0:
                print("Iteration: {:.3f}, Progress {:.2f}\n".format(iteration, iteration/len(test_loader)))

            #.........................................................................#
            res = model.forward(dataT.float())
            correctT += res.argmax(1).eq(targetT).sum().item()
            totalT += targetT.shape[0]
            lossT = criterion(res, targetT.long())

            y_np = targetT.cpu().numpy()
            y_pred_unrounded = np.append(y_pred_unrounded, res.cpu().numpy(), axis = 0)
            y_pred_np = res.argmax(1).cpu().numpy()
            y_pred = np.append(y_pred, y_pred_np)
            y_true = np.append(y_true, y_np)

        print('total: ' + str(totalT))
        print('Number correct: ' + str(correctT))
        print('accuracy= ' + str(correctT/totalT * 100))
        accuracyT = [correctT/totalT * 100]
        np.savetxt(dump_path + '/testAccuracy.csv', accuracyT, delimiter = ',')
        np.savetxt(dump_path + '/y_true.csv', y_true, delimiter = ',')
        np.savetxt(dump_path + '/y_pred.csv', y_pred, delimiter = ',')
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        #.........................................................................#
                             # save testing data
        
        keys = ['y_true', 'nPred', 'ePred']
        
        y_pred_unrounded_del = np.delete(y_pred_unrounded, 0, 0)
        neutron_preds = y_pred_unrounded_del[:, 0] #log softmax probabilities
        electron_preds = y_pred_unrounded_del[:, 1] #log softmax probabilities
        normalizedN = (neutron_preds-min(neutron_preds))/(max(neutron_preds)-min(neutron_preds))
        normalizedE = (electron_preds-min(electron_preds))/(max(electron_preds)-min(electron_preds))

        np.savetxt(dump_path + '/nPred.csv', normalizedN, delimiter = ',')
        np.savetxt(dump_path + '/ePred.csv', normalizedE, delimiter = ',')

