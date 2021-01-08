#
from dataLoaderWCH5 import WCH5Dataset, get_loaders
from training_utils import Train, Test
#
import argparse
import sys
import distutils.util
sys.path.insert(1, '/home/mattStubbs/github/Tools')
from models import *
from ioUtils import CSVData
import plotting_tools as ptool
from config import Config

def str2bool(v):
    return bool(distutils.util.strtobool(v))

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fc', type=str2bool, default=True,
                 help="Should the graph hits be fully connected?")
    parser.add_argument('--dyn', type=str2bool, default=False,
                 help="Dynamic graph?")
    parser.add_argument('--dw', type=str2bool, default=False,
                 help="distance weighted edges?")
    parser.add_argument('--model', type=str, default='GCN',
                help='network model (GCN, AGNN, SG or GAT)')
    parser.add_argument('--nfiles', type=str, default='6',
                help='number of files (6, 50 or ALL)')
    parser.add_argument('--epochs', type=int, default=5,
                help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                help='learning rate')
    parser.add_argument('--batch', type=int, default=32,
                help='batch size')
    parser.add_argument('--k', type=int, default=20,
                help='Number of nearest neighbours for not fully-connected graphs')
    parser.add_argument('--lrdecay', type=float, default=0.98,
                help='lerning rate decay rate')
    parser.add_argument('--print_interval', type=int, default=100,
                help='Interval to print the iteration number in terminal during training')
    parser.add_argument('--logNumber', type=int, default=1000,
                help='iteration interval for recording training results')
    parser.add_argument('--valNumber', type=int, default=10000,
                help='iteration interval for recording cross validation results')
    parser.add_argument('--trainWindow', type=int, default=10,
                help='moving average number for training plot display')
    parser.add_argument('--valWindow', type=int, default=3,
                help='moving average number for validation plot display')
    args = parser.parse_args()
    run(args)
    

def run(args):
    
    # .... config
    config = Config(args.model, args.nfiles, args.epochs, args.lr, args.batch)
    trainDL, valDL, testDL = get_loaders(config.data_path, config.train_indices_file, config.val_indices_file, config.test_indices_file, config.batch_size, 
                                         config.num_data_workers, k_neighbours=args.k, fully_connected=args.fc, dynamic=args.dyn, distance_weighted=args.dw)
                
    # .... model, train and test
    model_type = GCN()  # initialize model type
    if args.model.lower()=="gcn":
        model_type = GCN()
    if args.model.lower()=="gat":
        model_type = GAT()
    if args.model.lower()=="agnn":
        model_type = AGNN()
    print(model_type)
        
    model = Train(model_type, iteration_display=args.print_interval, log_number=args.logNumber, val_number=args.valNumber, train_loader=trainDL, val_loader=valDL, num_epochs=config.epochs, learning_rate=config.lr, dump_path=config.dump_path, lr_decay=args.lrdecay)
    Test(model=model_type, test_loader=testDL, dump_path=config.dump_path)
    
    # .... save meta-data
    with open(config.dump_path + '/metadata.txt', 'w') as writer:
        for param in vars(args):
            writer.write(str(param) + " " + str(getattr(args, param)) + "\n")

    
    # .... plotting
    class_labels = ('neutron', 'electron')
    ptool.plot_confusion_matrix(dump_path=config.dump_path+"/", classes=class_labels, model=args.model, fc=args.fc, dw=args.dw, k=args.k)
    ptool.disp_learn_hist_smoothed(location=config.dump_path+"/", window_train=args.trainWindow, window_val=args.valWindow, 
                                           model=args.model, fc=args.fc, dw=args.dw, k=args.k, show=False)
    ptool.ROC(location=config.dump_path+"/", model=args.model, fc=args.fc, dw=args.dw, k=args.k)
  

if __name__ == '__main__':
    main()
