from datetime import datetime
import os
class config(object):
    Project_DIR = "/media/hossam/Projects/NeedleDetection/"
    data_path = "/media/hossam/Projects/NeedleDetection/sample"
    """ """
    def __init__(self):
        self.Project_DIR = config.Project_DIR
        self.data_path = config.data_path
        self.save_images = self.Project_DIR + "cropped_saved_images"
        self.export_model = self.Project_DIR + "cropped_trained_models/"
        self.model_ckpt = self.export_model+"model.ckpt"
        self.logdir_train = self.Project_DIR + "tensorboard/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir_test = self.Project_DIR + "tensorboard/test/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.save_images):
            os.makedirs(self.save_images)
        if not os.path.exists(self.export_model):
            os.makedirs(self.export_model)
        """
        General settings
        """


        self.im_channels = 1
        self.orig_size_X = 1072
        self.orig_size_Y = 1381
        self.desired_size_X =  256
        self.desired_size_Y =  344
        self.desired_scale_X = 2
        self.desired_scale_Y = 2

        self.input_size_X = 512  # 1072//4
        self.input_size_Y = 688  # 1376//4
        self.input_size_half_X = 256
        self.input_size_half_Y = 344

        self.heatmap_size_X = 64 # 32#133//4
        self.heatmap_size_Y = 64 # 171//4
        self.numberOfHeatmaps = 1
        self.orig_scale_X = 2
        self.orig_scale_Y = 2
        self.scale_x = 1
        self.scale_y = 1
        self.stages = 2
        self.gaussian_variance = 0.5

        '''
        Training settings
        '''
        self.batch_size = 32
        self.output_node_names = 'PoseNet/Mconv5_stage2/BiasAdd'
        self.epochs = 512
        self.learningRate = 0.00002
        self.lr_step = 100
        self.feedbackPeriod = 50
        self.epochSaving = 3
