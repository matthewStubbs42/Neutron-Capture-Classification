from dataLoader_agg import *
from training_agg import train_features, Test
import argparse
import sys
import distutils.util
sys.path.insert(1, '/home/mattStubbs/github/Tools')
from models import MLP
from ioUtils import CSVData
import plotting_tools as ptool
from config import Config

def str2bool(v):
    return bool(distutils.util.strtobool(v))

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='MLP',
                help='network model (MLP or XGBoost)')
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
    args = parser.parse_args()
    run(args)
    

def run(args):
    
    # .... config
    config = Config(args.model, args.nfiles, args.epochs, args.lr, args.batch)
    print('loading dataset...')
    trainDL, valDL, testDL = get_loaders_features(config.data_path, config.train_indices_file, config.val_indices_file, config.test_indices_file, config.batch_size, config.num_data_workers)
    print('\ndone. loading model, beginning training...')        
    # .... model, train and test
    model_type = MLP()  # initialize model type
    model = train_features(model_type, iteration_display=args.print_interval, log_number=args.logNumber, val_number=args.valNumber, train_loader=trainDL, val_loader=valDL, num_epochs=config.epochs, learning_rate=config.lr, dump_path=config.dump_path, lr_decay=args.lrdecay)
    
    Test(model=model_type, test_loader=testDL, dump_path=config.dump_path)
    
    # .... save meta-data
    with open(config.dump_path + '/metadata.txt', 'w') as writer:
        writer.write(str(model_type))
        for param in vars(args):
            writer.write(str(param) + " " + str(getattr(args, param)) + "\n")
        
    # .... plotting
    class_labels = ('neutron', 'electron')
    ptool.plot_confusion_matrix(dump_path=config.dump_path+"/", classes=class_labels, model=args.model)
    ptool.disp_learn_hist_smoothed(location=config.dump_path+"/", window_train=args.trainWindow, window_val=args.valWindow, 
                                           model=args.model)
    ptool.ROC(location=config.dump_path+"/", model=args.model)

if __name__ == '__main__':
    main()

    