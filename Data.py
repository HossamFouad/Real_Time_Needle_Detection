import random
import os
import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter
import scipy
import pickle
import random
import cv2
import matplotlib
from skimage.transform import resize
from datetime import datetime
import math
from config import config
import matplotlib.pyplot as plt
def deNormalization_(im):
    im = np.asarray(im)
    im = np.float32(im)


    im = im*(255)
    im = np.asarray(im).astype(int)
    return im


def Normalization_(im):
    im = np.asarray(im)
    im = np.float32(im)

    maxI = np.max(im)
    minI = np.min(im)

    im = (im - minI) / (maxI - minI)
    im = np.asarray(im)
    return im

def produceCenterLabelMap(input_size_x,input_size_y,size, gaussian_variance=3, center=None):
    X = np.zeros((size, input_size_y), dtype=np.float32)
    x = np.arange(0, input_size_y, 1, float)
    X[np.arange(size),:]=x

    Y = np.zeros((size,input_size_x,1), dtype=np.float32)
    y = np.arange(0, input_size_x, 1, float)
    y = y[:, np.newaxis]

    Y[np.arange(size),:,:]=y

    x0 = center[:,0,:]
    y0 = center[:,1,:]
    X[:,:]=X-x0
    Y[:,:,0]= Y[:,:,0] - y0

    Z= [np.exp(-(X[i,:] ** 2 + Y[i,:,:] ** 2) / 2.0 / gaussian_variance / gaussian_variance) for i in range(size)]

    return np.stack(Z, axis=0)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class DataContainer:
    def __init__(self,type="train",FLAGS=config()):
        data_path = os.path.join(FLAGS.data_path, type)
        files = []
        self.FLAGS = FLAGS
        for r, d, f in os.walk(data_path):
            for file in f:
                if file.endswith(".npz"):
                    path = os.path.join(r, file)
                    y = np.load(path)['y']
                    if y[1] > FLAGS.orig_size_X - 48 or y[0] > FLAGS.orig_size_Y - 5:
                        print(path)
                        continue
                    files.append(os.path.join(r, file))

        #print(files)

        random.seed(datetime.now())
        random.shuffle(files)
        #print(files)
        self.size = len(files)
        print(self.size)
        self.X = np.zeros((self.size, FLAGS.input_size_X, FLAGS.input_size_Y,FLAGS.im_channels), dtype=np.float32)
        self.points = np.zeros((self.size, 2, 1),dtype=np.float32)
        if type=="testing":
            self.inferX = np.zeros((self.size, FLAGS.desired_size_X, FLAGS.desired_size_Y, FLAGS.im_channels),
                                   dtype=np.float32)

        i=-1
        self.m=0
        for f in files:

            if f.endswith(".npz"):
                i += 1
                x= np.load(f)['x']
                y =np.load(f)['y']
                x= rgb2gray(x)
                x = x[0:FLAGS.orig_size_X-48,0:FLAGS.orig_size_Y-5]
                if type == "testing":
                    test = scipy.misc.imresize(x, size=0.25)
                    self.inferX[i, :, :, 0] = Normalization_(test)

                x = scipy.misc.imresize(x, size=0.5)
                self.X[i,:,:,0] = Normalization_(x)
                y=[y]
                j=-1
                for m in y:
                    j+=1
                    m[0],m[1]=m[0]/FLAGS.orig_scale_X,m[1]/FLAGS.orig_scale_Y
                    self.points[i, :, :] =np.array([m[0],m[1]])[:,np.newaxis]

        if type == "testing":
           self.Xnew,self.points = self.cropping(self.X,self.points)
           #print(self.points)  # print(self.size)

           self.points[:, 0, :] = self.points[:, 0, :] // (FLAGS.scale_y)
           self.points[:, 1, :] = self.points[:, 1, :] //(FLAGS.scale_x)

           self.Y = np.zeros((np.shape(self.points)[0], FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y, FLAGS.numberOfHeatmaps),
                         dtype=np.float32)

           self.Y[:, :, :, 0] = produceCenterLabelMap(FLAGS.heatmap_size_X,FLAGS.heatmap_size_Y,np.shape(self.points)[0], gaussian_variance=FLAGS.gaussian_variance,
                                                  center=self.points)
        #print(np.shape(self.Y))
        #self.X= gaussian_filter(self.X, (0,4,4,0))
        #plt.imshow(self.X[7,:,:,0],cmap='gray', vmin=0, vmax=255)
        #print(np.shape(self.X), np.shape(self.Y))
        #print(self.points)
        #plt.imshow(1.0 * self.Y[1, :, :, 0] + 0.001 * scipy.misc.imresize(self.X[1, :, :, 0],
        #                                                           (FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y)))
        #plt.annotate('25, 50', xy=(y[0], y[1]), xycoords='data',xytext=(0.5, 0.5), textcoords='figure fraction',
        #             arrowprops=dict(arrowstyle="->"))
        #plt.show()
    def cropping(self,X,points):
        X_shape = np.shape(X)
        num =X_shape[0]
        new_X =  np.zeros((num, self.FLAGS.heatmap_size_X, self.FLAGS.heatmap_size_Y,self.FLAGS.im_channels), dtype=np.float32)
        new_points = np.zeros((num, 2, 1),dtype=np.float32)
        ind=[]
        for i in range(num):
            img = X[i]
            point = points[i]
            if point[1]<0 or point[0]<0:
                continue
            if point[1]> X_shape[2]or point[0]>X_shape[1]:
                continue
            xpoint = point[1].astype(int)[0]
            ypoint = point[0].astype(int)[0]
            xstart = xpoint-(self.FLAGS.heatmap_size_X)+8
            ystart = ypoint - (self.FLAGS.heatmap_size_Y )+8
            xcrop =random.randint(xstart, xpoint-16)
            ycrop = random.randint(ystart, ypoint-16)
            if xcrop<0:
                xcrop = 0
            elif xcrop+self.FLAGS.heatmap_size_X>X_shape[1]:
                print("x")
                continue
            if ycrop < 0:
                ycrop = 0
            elif ycrop + self.FLAGS.heatmap_size_Y > X_shape[2]:
                print("y")
                continue



            new_X[i, :, :, :] = img[xcrop:xcrop+self.FLAGS.heatmap_size_X,ycrop:ycrop+self.FLAGS.heatmap_size_Y,:]
            ind.append(i)
            new_points[i, 0, :] =points[i,0,:]-ycrop
            new_points[i, 1, :] = points[i, 1, :] - xcrop
        new_X = new_X[ind]
        new_points = new_points[ind]
        return new_X,new_points
    def get_pair(self):
        start = self.m%self.size
        end = (self.m+self.FLAGS.batch_size) % (self.size+1)
        self.m+=self.FLAGS.batch_size
        if(start>=end):
            start = 0
            end = self.FLAGS.batch_size
            self.m = self.FLAGS.batch_size
        return self.X[start:end].copy(),self.points[start:end].copy()



class NeedleData:
    def __init__(self,FLAGS=config()):
        self.FLAGS = FLAGS
        self.i=0
        self.train_data = DataContainer("train",FLAGS)
        self.train_size = self.train_data.size
        #print(self.train_size)
        self.test_data = DataContainer("testing",FLAGS)
        self.test_size = self.test_data.size

    def next_batch(self,augmented = True):
        X,point = self.train_data.get_pair()
        #print(point)
        if augmented:
            X,point =self.augment(X,point)

        point[:, 0, :] = point[:, 0, :] // (self.FLAGS.scale_y)
        point[:, 1, :] = point[:, 1, :] // (self.FLAGS.scale_x)
        X,point=  self.train_data.cropping(X, point)
        #print(np.shape(point)[0])
        Y = np.zeros((np.shape(point)[0], self.FLAGS.heatmap_size_X, self.FLAGS.heatmap_size_Y, self.FLAGS.numberOfHeatmaps),
                     dtype=np.float32)

        Y[:,:,:,0]= produceCenterLabelMap(self.FLAGS.heatmap_size_X, self.FLAGS.heatmap_size_Y,np.shape(point)[0], gaussian_variance=self.FLAGS.gaussian_variance,
                                                   center=point)
        #print(np.shape(X),np.shape(Y))
        return X, Y

    def get_test_data(self):
        start = self.i % self.test_size
        end = (self.i + self.FLAGS.batch_size) % (self.test_size+1)
        self.i+=self.FLAGS.batch_size
        if (start >= end):
            start = 0
            end = self.FLAGS.batch_size
            self.i = self.FLAGS.batch_size
        return self.test_data.Xnew[start:end],self.test_data.Y[start:end]

    def augment(self, X,points):
        factor = random.uniform(0.9, 1.1)
        rot = bool(random.getrandbits(1))
        zoomed = bool(random.getrandbits(1))
        smooth = bool(random.getrandbits(1))
        z = [1,1,1,1]
        axis = random.randint(1, 2)
        z[axis] = factor
        f2 = random.uniform(0.8, 1.2)
        X *= f2
        bin = [rot,zoomed,smooth]
        if bin[0]:
            print("rot")
            #angle_lst = [random.uniform(-5, 5)]
            #axes = random.choice([0,1,2,3])
            #print(axes)
            angle =random.uniform(-180, 180)# angle_lst[axes]
            X= rotate(X, angle=angle, axes=(2,1), reshape=False, order=1)
            #X = resize(X, (FLAGS.batch_size,FLAGS.input_size_X,FLAGS.input_size_Y,FLAGS.numberOfHeatmaps))
            angle = np.pi / 180 * angle
            m11 = math.cos(angle)
            m12 = math.sin(angle)
            m21 = -math.sin(angle)
            m22 = math.cos(angle)
            matrix = np.array([[m11, m12],
                                  [m21, m22]], dtype=np.float64)
            offset = np.zeros((2,), dtype=np.float64)
            offset[0] = float(self.FLAGS.input_size_Y) / 2.0 - 0.5
            offset[1] = float(self.FLAGS.input_size_X) / 2.0 - 0.5
            points -=offset[:,np.newaxis]
            points = np.matmul(matrix,points)
            points+=offset[:,np.newaxis]
        if bin[1]:
            print("zoom")
            factor = random.uniform(1, 1.2)
            z = [1, 1, 1,1]
            axis = random.randint(1, 2)
            #print(axis,factor)
            z[axis] = factor
            points[:, 2-axis, :] *=factor

            X=zoom(X, zoom=z, order=0)
            _,shapex,shapey,_ = np.shape(X)
            #print(np.shape(X))
            midx = shapex // 2
            midy = shapey // 2

            i = self.FLAGS.input_size_X//2
            j = self.FLAGS.input_size_Y//2

            X = X[:,midx-i:midx+i, midy-j:midy+j,:]

            offset = np.zeros((2,), dtype=np.float64)
            offset[1]=(shapex-self.FLAGS.input_size_X)/2.0
            offset[0] = (shapey - self.FLAGS.input_size_Y)/2.0
            points -= offset[:, np.newaxis]
        if True:
            print("sigma")
            sigma1 = random.uniform(0, 3)
            sigma2 = random.uniform(0, 3)
            X = gaussian_filter(X, (0,sigma1,sigma2,0))

        #print(points)
        return X,points


#print(os.listdir(os.path.join(os.getcwd(),"../sample","train")))
#X=NeedleData()
#x,y=X.get_test_data()
#for i in range(X.test_size):
 #   x,y=X.get_test_data()
  #  print(i)
   # for j in range(np.shape(x)[0]):
    #    matplotlib.image.imsave(os.path.join("/media/hossam/Projects/test", str(i*j) + '.png'),
     #                y[j, :, :, 0] * 1.0 + 1.2 * x[j, :, :, 0], cmap='gray')

#x,y=X.next_batch()
#x,y=X.next_batch()
#x,y=X.next_batch()
#img=deNormalization_(x[0, :, :, 0])
#img = np.stack([img, img, img], axis=-1)
#plt.imshow(img)
#plt.imshow(y[0, :, :, 0]*1.0+1.2*x[0, :, :, 0])
#plt.show()
#ax1=plt.subplot(2, 3, 1)
#ax1.imshow(1 * y[0, :, :, 0]+ 0.000 *scipy.misc.imresize(x[0, :, :, 0],(X.FLAGS.heatmap_size_X, X.FLAGS.heatmap_size_Y)))
#ax1=plt.subplot(2, 3, 4)
#ax1.imshow(img)
'''
ax1=plt.subplot(2, 3, 2)
ax1.imshow(1 * y[1, :, :, 0]+ 0.000 *scipy.misc.imresize(x[0, :, :, 0],(X.FLAGS.heatmap_size_X, X.FLAGS.heatmap_size_Y)))
ax1=plt.subplot(2, 3, 5)
img=deNormalization_(x[1, :, :, 0])
img = np.stack([img, img, img], axis=-1)
ax1.imshow(img)
ax1=plt.subplot(2, 3, 3)
ax1.imshow(1 * y[2, :, :, 0]+ 0.000 *scipy.misc.imresize(x[0, :, :, 0],(X.FLAGS.heatmap_size_X, X.FLAGS.heatmap_size_Y)))
ax1=plt.subplot(2, 3, 6)
img=deNormalization_(x[2, :, :, 0])
img = np.stack([img, img, img], axis=-1)
ax1.imshow(img)

'''

#plt.show()

#print(np.shape(x),np.shape(y))
#print(X.next_batch(0)[0])
#X,Y = X.get_pair(0)
#print(np.shape(Y))

'''
        z = scipy.misc.imresize(z, (FLAGS.orig_size_X, 1 * FLAGS.orig_size_Y))

        plt.imshow(z)
        #plt.imshow(0.0 * self.Y[0, :, :, 0] + 1.0 * scipy.misc.imresize(self.X[0, :, :, 0],
        #                                                           (FLAGS.heatmap_size_X, FLAGS.heatmap_size_Y)))
        y[0], y[1] = y[0] * FLAGS.orig_scale_Y*FLAGS.scale_x, y[1] * FLAGS.orig_scale_X*FLAGS.scale_x
        print(y)
        plt.annotate('25, 50', xy=(y[0], y[1]), xycoords='data',
                     xytext=(0.5, 0.5), textcoords='figure fraction',
                     arrowprops=dict(arrowstyle="->"))
        plt.show()

'''
#X=DataContainer("validation")
#print(np.random.randint(0,sel,50))
