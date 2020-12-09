#print(torch.cuda.is_available())
import copy
from ioUtils import CSVData
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

#...................................................................................................#
#...................................................................................................#
                                       # TRAINING
#...................................................................................................#
#...................................................................................................#

def Train(model, iteration_display, log_number, val_number, train_loader, val_loader, num_epochs, learning_rate, dump_path):
    
    #.........................................................................#
    train_log = CSVData(dump_path, "log_train.csv")
    val_log = CSVData(dump_path, "log_val.csv")
    best_val_log = CSVData(dump_path, "log_best_val.csv")
    keys = ['epoch', 'iteration', 'accuracy', 'loss']
    best_model_wts = copy.deepcopy(model.state_dict())

    #.........................................................................#
    epoch = 0.; iteration = 0; correct = 0.; acc = 0.; total = 0.; best_acc = 0.;
    best_acc = 0.;
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("dump path: {}".format(dump_path))
    #.........................................................................
    for epoch in range(num_epochs):
        # monitor training loss
        train_loss = 0.0 

        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1)); print('-' * 10)

        for data in train_loader:
            model.train()
            data = data.to(device) # send data to GPU 
            optimizer.zero_grad() # zero the parameter gradients
            if iteration % iteration_display == 0:  # display epoch and iteration
                print('epoch: {:.3f}, iteration: {:.2f}'.format(epoch, iteration))

            # forward pass
            output = model.forward(data) #compute output predictions
            loss = criterion(output, data.y) #compute loss from difference of predictions and actual labels
            loss.backward() #propagate loss backwards using gradient descent
            optimizer.step() #update parameters

            # training evaluation parameters
            correct += output.argmax(1).eq(data.y).sum().item()
            total += data.y.shape[0]
            acc = correct / total
            
            #.........................................................................#
                                    # update training log
                
            if iteration % log_number == 0 or iteration == 1 or iteration == len(train_loader)*num_epochs:
                train_log.record(keys, [epoch, iteration, acc, loss])
                train_log.write()
            
            #.........................................................................#
                                        # VALIDATION
                
            if iteration % val_number == 0 or iteration == len(train_loader)*num_epochs:
                model.eval()       # set to evaluation mode
                correctV = 0.; accV = 0.; totalV = 0.;
                
                for dataV in val_loader:  # iterate over validation dataloader
                
                    dataV = dataV.to(device)
                    outputV = model.forward(dataV) # compute output predictions
                    lossV = criterion(outputV, dataV.y) # compute loss 
                    
                    #evaluation parameters
                    correctV += outputV.argmax(1).eq(dataV.y).sum().item()
                    totalV += dataV.y.shape[0]
                
                accV = correctV / totalV
                
                if accV > best_acc:    # if validation accuracy better than best so far, update
                    best_acc = accV
                    print('{}accV: {:.4f}, bestAcc: {:.4f}{}'.format('-'*10, accV, best_acc, '-'*10))
                    best_val_log.record(keys, [epoch, iteration, accV, lossV])
                    best_val_log.write()
                    
                    #save best model parameters
                    torch.save(model.state_dict(), dump_path + "/" + "state_dict")
                    
                val_log.record(keys, [epoch, iteration, accV, lossV])
                val_log.write()
            
            #.........................................................................#
                                 # increment iteration, epoch
            iteration += 1
            epoch += 1 / len(train_loader)
            #.........................................................................#

        print('Epoch: {:.4f} Loss: {:.4f}, accuracy: {:.4f}'.format(epoch, loss, acc)) # printing update
    
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
    epoch=0.
    iteration=0
    criterion = nn.CrossEntropyLoss()

    #.........................................................................#
    with torch.no_grad():           

        correctT = 0.; totalT = 0.; accT = 0.; lossT = 0.;
        y_pred = []; y_true = []; y_pred_np = []
        y_pred_unrounded = np.zeros([1,2])
        
        #.........................................................................#
        for dataT in test_loader:

            dataT = dataT.to(device)
            iteration += 1

            if iteration % 200 == 0:
                print("Iteration: {:.3f}, Progress {:.2f}\n".format(iteration, iteration/len(testDL)))

            #.........................................................................#
            res = model.forward(dataT)
            correctT += res.argmax(1).eq(dataT.y).sum().item()
            totalT += dataT.y.shape[0]
            lossT = criterion(res, dataT.y)

            y_np = dataT.y.cpu().numpy()
            y_pred_unrounded = np.append(y_pred_unrounded, res.cpu().numpy(), axis = 0)
            y_pred_np = res.argmax(1).cpu().numpy()
            y_pred = np.append(y_pred, y_pred_np)
            y_true = np.append(y_true, y_np)

        print('total: ' + str(totalT))
        print('Number correct: ' + str(correctT))
        print('accuracy= ' + str(correctT/totalT * 100))
        accuracyT = [correctT/totalT * 100]
#         print('y_pred: ' + str(y_pred))
#         print('y_actual: ' + str(y_true))
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

        