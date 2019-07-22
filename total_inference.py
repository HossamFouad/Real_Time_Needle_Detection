import cv2
import numpy as np
#fs = cv2.FileStorage("/home/hossam/img.ext", cv2.FILE_STORAGE_FORMAT_YAML)

#fn = fs.getNode("Tracking").mat()
fn = np.load("/home/hossam/inference.npz")['x']
print(np.shape(fn[1:,:]))
np.savez("/home/hossam/inference1.npz",x=fn[1:,:])
'''
import matplotlib.pyplot as plt
plt.ion()
lst=[]
y = []

for i in range(4):
    lst.append(i)
    y.append(fn[0][0])
    fig = plt.figure()
    ax = plt.subplot(111)
   
    ax.plot(lst, y, label='$y = numbers')
    plt.title('Legend inside')
    ax.legend()
    fig.savefig('/home/hossam/'+str(i)+'plot.png')
'''
