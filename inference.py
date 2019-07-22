import sys
import numpy as np
import tensorflow as tf
import Model
from config import config
import matplotlib.pyplot as plt
from Data import DataContainer
import CostFunction
import scipy
import cv2
import matplotlib
import os
import warnings
import argparse
import time
try:
    from polyaxon_helper import get_data_paths
    config.data_path = os.path.join(get_data_paths()['data1'], 'Micromanipulator/NeedleTracking3/40xdatasets')
except:
    warnings.warn("You are Running on the local Machine", UserWarning)

    #print(os.listdir(get_data_paths()['data1']))
    #print(get_data_paths()[0]  )
print()
#print(config.data_path )
parser = argparse.ArgumentParser()
parser.add_argument('--out', default='/media/hossam/Projects/NeedleDetection/')
args = parser.parse_args()
config.Project_DIR = vars(args)['out']+"/"
FLAGS = config()
FLAGS.batch_size = 2

def deNormalization_(im):
    im = np.asarray(im)
    im = np.float32(im)

    im = im*(255)
    im = np.asarray(im).astype(np.uint8)
    return im
class infer:
    def __init__(self):
        self.FLAGS =  FLAGS
        self.i=-FLAGS.batch_size
        self.test_data = DataContainer("testing",FLAGS)
        print(self.test_data.size)
    def get_test_data(self,inferenceFlag=None):
        start = self.i % self.test_data.size
        end = (self.i + self.FLAGS.batch_size) % (self.test_data.size+1)
        self.i+=self.FLAGS.batch_size
        if (start >= end):
            start = 0
            end = self.FLAGS.batch_size
            self.i = self.FLAGS.batch_size
        if inferenceFlag is not None:
            return self.test_data.X[start:end],self.test_data.Y[start:end],self.test_data.inferX[start:end]
        else:
            return self.test_data.Xnew[start:end],self.test_data.Y[start:end]

        '''
        self.i+=self.FLAGS.batch_size

        if self.i >= self.test_data.size:
            self.i=0

        return self.test_data.X[self.i:self.i+self.FLAGS.batch_size],self.test_data.Y[self.i:self.i+self.FLAGS.batch_size]
        '''
def inference():
    data = infer()

    with tf.name_scope('input'):
        inputImage = tf.placeholder(tf.float32, shape=[None, FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y, FLAGS.im_channels],
                                    name="input-image")
        gtHeatmap = tf.placeholder(tf.float32,
                                   shape=[None, FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y, FLAGS.numberOfHeatmaps],
                                   name="gt-heatmaps")

    estimatedHeatmaps = Model.inference_pose(inputImage, FLAGS.numberOfHeatmaps)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_ckpt)
        test_iterations = np.ceil(data.test_data.size / FLAGS.batch_size).astype(int)

        print('Inference')
        num = 0
        for i in range(test_iterations):
            X_valid, Y_valid = data.get_test_data()
            train_dict = {inputImage: X_valid, gtHeatmap: Y_valid}
            start = time.time()
            vout = sess.run([estimatedHeatmaps], feed_dict=train_dict)
            elapsed = time.time()
            elapsed = elapsed - start
            print("Time spent in (function name) is: ", elapsed*1000)


            prediction = vout[0][FLAGS.stages - 1][:,:,:,:]
            Pshape = np.shape(prediction)
            Ishape = np.shape(X_valid)
            #Ishape = np.shape(inferX)

            for j in range(Pshape[0]):
                print("image num {}...".format(num))
                for k in range(FLAGS.numberOfHeatmaps):
                    heatmap_pred = scipy.misc.imresize(prediction[j, :, :, k], size=(Ishape[1], Ishape[2]))
                    heatmap_GT = scipy.misc.imresize(Y_valid[j, :, :, k], size=(Ishape[1], Ishape[2], Ishape[3]))
                    ind_pred = np.unravel_index(np.argmax(heatmap_pred, axis=None), heatmap_pred.shape)
                    ind_GT = np.unravel_index(np.argmax(heatmap_GT, axis=None), heatmap_GT.shape)
                    img = deNormalization_(X_valid[j,:,:,0])
                    matplotlib.image.imsave(os.path.join(FLAGS.save_images, str(num) + '.png'), img, cmap='gray')
                    #img = deNormalization_(inferX[j,:,:,0])

                    img = np.stack([img, img, img], axis=-1)
                    img = cv2.drawMarker(img, (ind_pred[1], ind_pred[0]), (255, 0, 0),
                                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2,
                                       line_type=cv2.LINE_AA)
                    img = cv2.drawMarker(img, (ind_GT[1], ind_GT[0]), (0, 0, 255),
                                         markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2,
                                         line_type=cv2.LINE_AA)
                    #img = cv2.rectangle(img, (ind_pred[1]-32, ind_pred[0]-32), (ind_pred[1]+32, ind_pred[0]+32), (255, 0, 0), 3)
                    pix_errorX = np.abs(ind_pred[0] - ind_GT[0])
                    pix_errorY = np.abs(ind_pred[1] - ind_GT[1])
                    print("pixel error for landmark {} = ({},{})".format(k,pix_errorX,pix_errorY))
                img = cv2.rectangle(img, (FLAGS.input_size_Y-125, 10), (FLAGS.input_size_Y-10, 60), (0, 0, 0), 1)
                img = cv2.putText(img, "Ground Truth: X", (FLAGS.input_size_Y-120, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                img = cv2.putText(img, "Estimated: X", (FLAGS.input_size_Y - 120, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0),
                                  lineType=cv2.LINE_AA)

                matplotlib.image.imsave(os.path.join(FLAGS.save_images, str(num)+'.png'), img, cmap='gray')
                num+=1

def main():
    inference()

main()