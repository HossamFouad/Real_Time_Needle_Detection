import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import functools

def inference_pose(image, numHM):
  # corresponds to pose_deploy_centerMap.prototxt

  with tf.variable_scope('PoseNet'):
    w = 8
    conv1_stage1 = layers.conv2d(image, w*4, 9, 1, activation_fn=None, scope='conv1_stage1')
    conv1_stage1 = tf.nn.relu(conv1_stage1)
    conv2_stage1 = layers.conv2d(conv1_stage1, w*4, 9, 1, activation_fn=None, scope='conv2_stage1')
    conv2_stage1 = tf.nn.relu(conv2_stage1)
    conv3_stage1 = layers.conv2d(conv2_stage1, w*4, 9, 1, activation_fn=None, scope='conv3_stage1')
    conv3_stage1 = tf.nn.relu(conv3_stage1)
    conv4_stage1 = layers.conv2d(conv3_stage1, w, 5, 1, activation_fn=None, scope='conv4_stage1')
    conv4_stage1 = tf.nn.relu(conv4_stage1)
    conv5_stage1 = layers.conv2d(conv4_stage1, w*16, 9, 1, activation_fn=None, scope='conv5_stage1')
    conv5_stage1 = tf.nn.relu(conv5_stage1)
    conv6_stage1 = layers.conv2d(conv5_stage1, w*16, 1, 1, activation_fn=None, scope='conv6_stage1')
    conv6_stage1 = tf.nn.relu(conv6_stage1)
    conv7_stage1 = layers.conv2d(conv6_stage1, numHM, 1, 1, activation_fn=None, scope='conv7_stage1')


    conv1_stage2 = layers.conv2d(image, w*4, 9, 1, activation_fn=None, scope='conv1_stage2')
    conv1_stage2 = tf.nn.relu(conv1_stage2)
    conv2_stage2 = layers.conv2d(conv1_stage2, w*4, 9, 1, activation_fn=None, scope='conv2_stage2')
    conv2_stage2 = tf.nn.relu(conv2_stage2)
    conv3_stage2 = layers.conv2d(conv2_stage2, w*4, 9, 1, activation_fn=None, scope='conv3_stage2')
    conv3_stage2 = tf.nn.relu(conv3_stage2)
    conv4_stage2 = layers.conv2d(conv3_stage2, w, 5, 1, activation_fn=None, scope='conv4_stage2')
    conv4_stage2 = tf.nn.relu(conv4_stage2)

    concat_stage2 = tf.concat(axis=3, values=[conv4_stage2, conv7_stage1])

    Mconv1_stage2 = layers.conv2d(concat_stage2, w*4, 11, 1, activation_fn=None, scope='Mconv1_stage2')
    Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
    Mconv2_stage2 = layers.conv2d(Mconv1_stage2, w*4, 11, 1, activation_fn=None, scope='Mconv2_stage2')
    Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
    Mconv3_stage2 = layers.conv2d(Mconv2_stage2, w*4, 11, 1, activation_fn=None, scope='Mconv3_stage2')
    Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
    Mconv4_stage2 = layers.conv2d(Mconv3_stage2, w*4, 1, 1, activation_fn=None, scope='Mconv4_stage2')
    Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
    Mconv5_stage2 = layers.conv2d(Mconv4_stage2, numHM, 1, 1, activation_fn=None, scope='Mconv5_stage2')
    '''

    conv1_stage3 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None, scope='conv1_stage3')
    conv1_stage3 = tf.nn.relu(conv1_stage3)

    concat_stage3 = tf.concat(axis=3, values=[conv1_stage3, Mconv5_stage2])

    Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 11, 1, activation_fn=None, scope='Mconv1_stage3')
    Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
    Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 11, 1, activation_fn=None, scope='Mconv2_stage3')
    Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
    Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 11, 1, activation_fn=None, scope='Mconv3_stage3')
    Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
    Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 1, 1, activation_fn=None, scope='Mconv4_stage3')
    Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
    Mconv5_stage3 = layers.conv2d(Mconv4_stage3, numHM, 1, 1, activation_fn=None, scope='Mconv5_stage3')
    
    conv1_stage4 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None, scope='conv1_stage4')
    conv1_stage4 = tf.nn.relu(conv1_stage4)

    concat_stage4 = tf.concat(axis=3, values=[conv1_stage4, Mconv5_stage3])

    Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 11, 1, activation_fn=None, scope='Mconv1_stage4')
    Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
    Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 11, 1, activation_fn=None, scope='Mconv2_stage4')
    Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
    Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 11, 1, activation_fn=None, scope='Mconv3_stage4')
    Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
    Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 1, 1, activation_fn=None, scope='Mconv4_stage4')
    Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
    Mconv5_stage4 = layers.conv2d(Mconv4_stage4, numHM, 1, 1, activation_fn=None, scope='Mconv5_stage4')
    
    conv1_stage5 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None, scope='conv1_stage5')
    conv1_stage5 = tf.nn.relu(conv1_stage5)

    concat_stage5 = tf.concat(axis=3, values=[conv1_stage5, Mconv5_stage4])

    Mconv1_stage5 = layers.conv2d(concat_stage5, 128, 11, 1, activation_fn=None, scope='Mconv1_stage5')
    Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
    Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 128, 11, 1, activation_fn=None, scope='Mconv2_stage5')
    Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
    Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 128, 11, 1, activation_fn=None, scope='Mconv3_stage5')
    Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
    Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 128, 1, 1, activation_fn=None, scope='Mconv4_stage5')
    Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
    Mconv5_stage5 = layers.conv2d(Mconv4_stage5, numHM, 1, 1, activation_fn=None, scope='Mconv5_stage5')

    conv1_stage6 = layers.conv2d(pool3_stage2, 32, 5, 1, activation_fn=None, scope='conv1_stage6')
    conv1_stage6 = tf.nn.relu(conv1_stage6)

    concat_stage6 = tf.concat(axis=3, values=[conv1_stage6, Mconv5_stage5])
    Mconv1_stage6 = layers.conv2d(concat_stage6, 128, 11, 1, activation_fn=None, scope='Mconv1_stage6')
    Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
    Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 128, 11, 1, activation_fn=None, scope='Mconv2_stage6')
    Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
    Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 128, 11, 1, activation_fn=None, scope='Mconv3_stage6')
    Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
    Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 128, 1, 1, activation_fn=None, scope='Mconv4_stage6')
    Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
    Mconv5_stage6 = layers.conv2d(Mconv4_stage6, numHM, 1, 1, activation_fn=None, scope='Mconv5_stage6')
    '''
    #heatmaps = [conv7_stage1, Mconv5_stage2, Mconv5_stage3, Mconv5_stage4, Mconv5_stage5, Mconv5_stage6]
    heatmaps = [conv7_stage1 , Mconv5_stage2]#, Mconv5_stage3]

  return heatmaps