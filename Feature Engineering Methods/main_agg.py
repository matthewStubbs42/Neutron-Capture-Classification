from dataLoader_agg import *
from training_agg import train_MLP, Test_XGB, Test_MLP
import argparse
import sys
import distutils.util
sys.path.insert(1, '/home/mattStubbs/github/Tools')
from models import MLP
from ioUtils import CSVData
import plotting_tools as ptool
from config import Config
import os

def str2bool(v):
    return bool(distutils.util.strtobool(v))

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='MLP',
                help='network model (MLP or XGB)')
    parser.add_argument('--nfiles', type=str, default='ALL',
                help='number of files (ALL or dn)')
    parser.add_argument('--epochs', type=int, default=5,
                help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.003,
                help='learning rate')
    parser.add_argument('--batch', type=int, default=512,
                help='batch size')
    parser.add_argument('--print_interval', type=int, default=100,
                help='Interval to print the iteration number in terminal during training')
    parser.add_argument('--logNumber', type=int, default=200,
                help='iteration interval for recording training results')
    parser.add_argument('--valNumber', type=int, default=5000,
                help='iteration interval for recording cross validation results')
    parser.add_argument('--trainWindow', type=int, default=7,
                help='moving average number for training plot display')
    parser.add_argument('--valWindow', type=int, default=1,
                help='moving average number for validation plot display')
    parser.add_argument('--lrdecay', type=float, default=1.,
                help='lerning rate decay rate')
    parser.add_argument('--depth', type=int, default=5,
                help='max XGBoost tree depth')
    parser.add_argument('--child_weight', type=int, default=3,
                help='minimum XGBoost tree child weight')
    parser.add_argument('--subsample', type=float, default=0.8,
                help='XGBoost tree subsample amount')
    parser.add_argument('--c1', type=str2bool, default=False,
                       help="Configuration 1: beta parameters only b1-b5")
    args = parser.parse_args()
    run(args)
    

def run(args):
    
    # .... config
    dn_flag = False
    if args.nfiles.lower()=="dn":
        dn_flag = True
    config = Config(args.model, args.nfiles, args.epochs, args.lr, args.batch)
    print(config.dump_path)
    log = CSVData(config.dump_path, "log_train.csv")
    
    # .... model, train and test for MLP
    if config.model_name == 'MLP':
        if args.c1:
            model_type = MLP(c1=True)
        else:
            model_type = MLP()
        # get dataloaders
        print(model_type.inputs)
        trainDL, valDL, testDL = get_loaders_features(config.data_path, config.train_indices_file, config.val_indices_file, config.test_indices_file, config.batch_size, config.num_data_workers,c1=args.c1, dn=dn_flag)
        # train model
        model = train_MLP(model_type, iteration_display=args.print_interval, log_number=args.logNumber, val_number=args.valNumber, train_loader=trainDL, val_loader=valDL, num_epochs=config.epochs, learning_rate=config.lr, dump_path=config.dump_path, lr_decay=args.lrdecay)
        # evaluate model
        Test_MLP(model=model_type, test_loader=testDL, dump_path=config.dump_path)
        
    elif config.model_name == 'XGB':
        # get XGB datasets
        xgb_data = XGB_Dataset(config.data_path, config.train_indices_file, config.val_indices_file, config.test_indices_file)
        dtrain, dval, dtest = xgb_data.dtrain, xgb_data.dval, xgb_data.dtest
        evals, evals_result = [(dtrain,'train'), (dval,'val')], {}
        # set training parameters
        params = {'max_depth':args.depth, 'min_child_weight':args.child_weight, 'learning_rate':args.lr, 'subsample':args.subsample, 'eval_metric' : ['logloss']}
        # train model
        bst = xgb.train(params, dtrain, evals=evals, num_boost_round=config.epochs, evals_result = evals_result)  
        os.mkdir(config.dump_path)
        bst.save_model(config.dump_path + "/best.model")
        # evaluate model
        Test_XGB(config.dump_path, dtrain, dval, dtest, evals_result)
            
    # .... save meta-data
    with open(config.dump_path + '/metadata.txt', 'w') as writer:
        writer.write(str(config.model_name))
        for param in vars(args):
            writer.write(str(param) + " " + str(getattr(args, param)) + "\n")
        
    # .... plotting
    class_labels = ('neutron', 'electron')
    if config.model_name != "XGB":
        ptool.plot_confusion_matrix(dump_path=config.dump_path+"/", classes=class_labels, model=args.model)
        ptool.disp_learn_hist_smoothed(location=config.dump_path+"/", window_train=args.trainWindow, window_val=args.valWindow, 
                                               model=args.model)
        ptool.ROC(location=config.dump_path+"/", model=args.model)
        

if __name__ == '__main__':
    main()

    