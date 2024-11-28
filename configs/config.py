import os
class Configs():
    def __init__(self):
        ## Intitation
        self.MODEL_NAME = "SiameseNetwork"
        self.LOSS = "ContrastiveLoss"
        self.OPTIMIZER = "Adam"

        ## Directory 
        self.DATA_DIR = "/media/baihaqi/Data_Temp/SiameseNetwork/Data/exp_1/"
        self.NUM_CLASSES = len(os.listdir(self.DATA_DIR))

        ## HYPERPARAMETERS  TRAINING
        self.BATCH_SIZE = 32
        self.DEVICE = "cuda:0"
        self.NUM_WORKERS = 4
        self.NUM_EPOCHS = 100
        self.TRAIN_BATCH_SIZE = 32
        self.VAL_BATCH_SIZE = 32
        self.LEARNING_RATE = 0.0001
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 5E-4
        self.SCHEDULER_STEP_SIZE = 10
        self.SCHEDULER_GAMMA = 0.5
        self.SCHEDULER = "ReduceLROnPlateau"
        self.EARLY_STOP = "ReduceLROnPlateau"
        self.EARLY_STOP_MIN_DELTA = 0
        self.EARLY_STOP_PATIENCE = 10
        self.MODEL = "SiameseNetwork"

        ## Training
        
        self.CHECKPOINT_INTERVAL = 20
