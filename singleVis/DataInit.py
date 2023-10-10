import torch
import sys
sys.path.append("..")
from singleVis.data import NormalDataProvider
from scipy.special import softmax
import numpy as np
import os
import json
from singleVis.utils import *

class DataInit():
    def __init__(self, config_path, content_path, cur_epoch, DEVICE):
        self.content_path = content_path
        self.config_path = config_path
        self.cur_epoch = cur_epoch
        self.model_path = os.path.join(self.content_path, "Model")
        self.DEVICE = DEVICE

    def get_conf(self, predction):
        scores = np.amax(softmax(predction, axis=1), axis=1)
        return scores

    def getData(self):
        sys.path.append(self.content_path)
        sys.path.append(self.config_path)

        with open(os.path.join(self.config_path,'config.json'), 'r') as f:
            config = json.load(f)
        ####### parameter ######
        TRAINING_PARAMETER = config["TRAINING"]
        NET = TRAINING_PARAMETER["NET"]
        CLASSES = config["CLASSES"]
        GPU_ID = config["GPU"]
        EPOCH_START = config["EPOCH_START"]
        EPOCH_END = config["EPOCH_END"]
        EPOCH_PERIOD = config["EPOCH_PERIOD"]
        DEVICE = self.DEVICE

        LEN = TRAINING_PARAMETER["train_num"]
        # Training parameter (visualization model)
        VISUALIZATION_PARAMETER = config["VISUALIZATION"]
        L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
        print('NET',NET)
        import Model.model as subject_model
        net = eval("subject_model.{}()".format(NET))

        # sys.path.remove()
        sys.path.remove(self.content_path)

        data_provider = NormalDataProvider(self.content_path, net,EPOCH_START, EPOCH_END, EPOCH_PERIOD,  device=DEVICE, epoch_name='Epoch',classes=CLASSES,verbose=1)
        train_data = data_provider.train_representation(self.cur_epoch).squeeze()
        prediction = data_provider.get_pred(self.cur_epoch, train_data)
        prediction_label = prediction.argmax(axis=1)
        confidence = self.get_conf(prediction)


        return net, data_provider, train_data, prediction, prediction_label, confidence
    
    def get_boundary_point(self, model, DEVICE,l_bound,num_adv_eg):
        ## Loading training data
        training_data_path = os.path.join(self.content_path, "Training_data")
        training_data = torch.load(os.path.join(training_data_path, "training_dataset_data.pth"),
                                   map_location=DEVICE)
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(self.cur_epoch), "index.json")
        index = load_labelled_data_index(index_file)
        training_data = training_data[index]
        ####

        confs = batch_run(model, training_data)
        preds = np.argmax(confs, axis=1).squeeze()
        border_points, cur_sel, _ = get_border_points(model=model, input_x=training_data, confs=confs, predictions=preds, device=DEVICE, l_bound=l_bound, num_adv_eg=num_adv_eg, lambd=0.05, verbose=0)
        return border_points, cur_sel

        






 
