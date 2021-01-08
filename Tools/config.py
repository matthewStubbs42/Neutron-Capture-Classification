import datetime

##......................................................................................................##
#                                        Config Settings                                                 #
##......................................................................................................##

class Config:
    def __init__(self, model_name, nfiles, epochs, lr, batch_size, device="cuda", gpu_list=[0], num_workers=0):
        
        ##......................................................................................................##
        h5_filepath = "/home/mattStubbs/watchmal/NeutronGNN/data/h5_files/iwcd_mpmt_shorttank_neutrongnn_" + nfiles + ".h5"
        train_indices_file = "/home/mattStubbs/watchmal/NeutronGNN/data/splits/train_indicies_" + nfiles + ".txt"
        val_indices_file = "/home/mattStubbs/watchmal/NeutronGNN/data/splits/validation_indicies_" + nfiles + ".txt"
        test_indices_file = "/home/mattStubbs/watchmal/NeutronGNN/data/splits/test_indicies_" + nfiles + ".txt"
        dump_path = "/home/mattStubbs/watchmal/NeutronGNN/data/dump/" + model_name + "/" + nfiles + "/" + str(datetime.datetime.now())[:-7]
        ##......................................................................................................##
        
        self.data_path = h5_filepath
        self.train_indices_file = train_indices_file
        self.val_indices_file = val_indices_file
        self.test_indices_file = test_indices_file

        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.gpu_list = gpu_list
        self.num_data_workers = num_workers
        self.epochs = epochs
        self.model_name = model_name
        self.dump_path = dump_path
 