import sys
import numpy as np
import tensorflow as tf
import Model
from config import config
import matplotlib.pyplot as plt
from Data import NeedleData
import CostFunction
import scipy
import argparse
import os
import warnings
cluster = True
"""

"""

try:
    from polyaxon_helper import get_data_paths
    config.data_path = os.path.join(get_data_paths()['data1'], 'Micromanipulator/NeedleTracking4')
except:
    warnings.warn("You are Running on the local Machine", UserWarning)

#print(os.listdir(get_data_paths()['data1']+"/Micromanipulator/NeedleTracking3"))
    #print(get_data_paths()[0]  )

#print(config.data_path )
parser = argparse.ArgumentParser()
parser.add_argument('--out', default='/media/hossam/Projects/NeedleDetection/')
args = parser.parse_args()
config.Project_DIR = vars(args)['out']+"/"
FLAGS = config()

def train(restore = False):
    tf.reset_default_graph()
    data = NeedleData(FLAGS=FLAGS)

    with tf.name_scope('input'):
        inputImage = tf.placeholder(tf.float32,shape=[None,FLAGS.heatmap_size_X,FLAGS.heatmap_size_Y, FLAGS.im_channels], name="input-image")
        gtHeatmap = tf.placeholder(tf.float32,shape=[None, FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y, FLAGS.numberOfHeatmaps], name="gt-heatmaps")


    estimatedHeatmaps = Model.inference_pose(inputImage, FLAGS.numberOfHeatmaps)

    cost, individualCosts, heatMapCosts = CostFunction.calculateCost(estimatedHeatmaps, gtHeatmap, FLAGS.numberOfHeatmaps)
    tf.summary.scalar('costfunction', cost / FLAGS.batch_size)
    for i in range(0, len(individualCosts)):
        tf.summary.scalar('costfunctionStage' + str(i), individualCosts[i] / FLAGS.batch_size)
    for i in range(0, len(heatMapCosts)):
        tf.summary.scalar('costfunctionHeatMap' + str(i), heatMapCosts[i] / FLAGS.batch_size)


    #tf.summary.image("Data", tf.expand_dims(estimatedHeatmaps[2][:,:,:,0],-1) * 0.9 + 0.1 * tf.image.resize_images(inputImage, (
    #FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y)))

    merged = tf.summary.merge_all()

    merged_test =tf.summary.merge([tf.summary.image("Testing data", estimatedHeatmaps[FLAGS.stages-1] * 0.9 + 0.1 * tf.image.resize_images(inputImage,
                                    (FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y)))])


    # Optimizer
    opt = tf.train.AdamOptimizer(FLAGS.learningRate).minimize(cost)

    # Buffer to save intermediate results
    lossesInPeriod = []
    lossesInPeriodTf = tf.placeholder(tf.float32, name="mean_loss")
    averagedCost = tf.reduce_mean(tf.reshape(lossesInPeriodTf, shape=[FLAGS.feedbackPeriod]))
    iterations = np.ceil(data.train_size / FLAGS.batch_size).astype(int)
    test_iterations = np.ceil(data.test_size / FLAGS.batch_size).astype(int)
    #print(iterations,test_iterations)
    with tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.logdir_train,sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.logdir_test, sess.graph)
        saver = tf.train.Saver()
        if restore:
            saver.restore(sess, FLAGS.model_ckpt)
        def compute_val_loss(epoch):
            vl = 0
            print('Validation after epoch: {}'.format(epoch))
            #print(test_iterations)
            for i in range(test_iterations):
                X_valid, Y_valid = data.get_test_data()
                train_dict = {inputImage: X_valid, gtHeatmap: Y_valid}
                [summary1,summary,val_loss, vout] = sess.run([merged_test,merged,cost, estimatedHeatmaps],
                                            feed_dict=train_dict)
                test_writer.add_summary(summary,i)
                test_writer.add_summary(summary1, i)
                vl+=val_loss
            vl/=data.test_size
            print('Validation loss: {}, '.format(vl))
            return vl


        min_val_loss = compute_val_loss(-1)

        for epoch in range(FLAGS.epochs):
            print("Epoch: ",epoch)
            for b in range(iterations):
                vout = []
                X_train,Y_train = data.next_batch()
                train_dict = {inputImage: X_train, gtHeatmap: Y_train}
                sess.run(opt,feed_dict = train_dict)

                [summary, Tcost, vout] = sess.run([merged,cost, estimatedHeatmaps],
                                            feed_dict=train_dict)
                #print(np.shape(vout))
                print('Batch: {}, loss: {} '.format(b, Tcost/FLAGS.batch_size))
                train_writer.add_summary(summary,b)

            val_loss = compute_val_loss(epoch)
            #z = scipy.misc.imresize(X_train[0, :, :, 0], (FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y))
            #plt.imshow(z)
            #plt.imshow(1.0 * vout[2][0,:,:,0] + 0.001 * z)
            #plt.show()

            if val_loss<min_val_loss:
                saver.save(sess,FLAGS.model_ckpt)
                print("Saving Model ... ")
                min_val_loss=val_loss



def save_model():
    tf.reset_default_graph()

    with tf.name_scope('input'):
        inputImage = tf.placeholder(tf.float32, shape=[None, FLAGS.input_size_X, FLAGS.input_size_Y, FLAGS.im_channels],
                                    name="input-image")
        gtHeatmap = tf.placeholder(tf.float32,
                                   shape=[None, FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y, FLAGS.numberOfHeatmaps],
                                   name="gt-heatmaps")

    estimatedHeatmaps = Model.inference_pose(inputImage, FLAGS.numberOfHeatmaps)



    with tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_ckpt)
        tf.train.write_graph(sess.graph_def,FLAGS.model_ckpt , FLAGS.export_model+'saved_model.pbtxt')
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = FLAGS.output_node_names
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                    output_node_names.split(","))
        output_graph = FLAGS.export_model+"CroppedNeedle.pb"

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

def main():
    train()
    save_model()

main()
