from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import tensorflow as tf

from common import estimate_pose, draw_humans, read_imgfile

import time

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimage', type=str)
    parser.add_argument('--nimage', type=str)
    args = parser.parse_args()

def inference(image):
	t0 = time.time()
	tf.reset_default_graph()
	from tensorflow.core.framework import graph_pb2
	graph_def = graph_pb2.GraphDef()
    # Download model from https://www.dropbox.com/s/2dw1oz9l9hi9avg/optimized_openpose.pb
	with open('models/optimized_openpose.pb', 'rb') as f:
		graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')

    
	inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
	heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
	pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

    
	with tf.Session() as sess:
		heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
			inputs: image
			})        

		heatMat, pafMat = heatMat[0], pafMat[0]

		humans = estimate_pose(heatMat, pafMat)
		arr = [v[1] for v in humans[0].values()]
		h = np.array(arr,'d')	
		return h
        



def pos_comp(model,noob):
	m1,m2,disparity = procrustes(model, noob)
	print (disparity)


# a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
# b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
# plt.plot(a,b)
# plt.show
a = inference(read_imgfile(args.mimage, 656, 368))
b = inference(read_imgfile(args.nimage, 656, 368))
# pos_comp(a,b)
sa = a.size/2
sb = b.size/2

if sa > sb:
	a = a[0:int(sb)]
if sb > sa:
	b = b[0:int(sa)]

pos_comp(a,b)

