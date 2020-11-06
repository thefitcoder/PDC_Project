def processFrame(path_to_ckpt,x,q):
	import time
	start=time.time()
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
	print('tensorflow imported in time:', time.time()-start)
	start=time.time()
	import multiprocessing
	import cv2
	import numpy as np
	path_to_ckpt = path_to_ckpt
	
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	default_graph = detection_graph.as_default()
	print('Model loaded in time:', time.time()-start)
	start=time.time()
	sess = tf.Session(graph=detection_graph)
	# Definite input and output Tensors for detection_graph
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	# Each box represents a part of the image where a particular object was detected.
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')
	# Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
	d=0
	print('Session started in time:', time.time()-start)
	while True:
		if not x.empty() and q.empty():
			image=x.get()
			print('recieved from x Queue')
			image_np_expanded = np.expand_dims(image, axis=0)
			# Actual detection.
			print('starting prediction')
			start_time = time.time()
			(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})
			end_time = time.time()
			print("FPS of predictor:", 1/(end_time-start_time))
			im_height, im_width,_ = image.shape
			boxes_list = [None for i in range(boxes.shape[1])]
			for i in range(boxes.shape[1]):
				boxes_list[i] = (int(boxes[0,i,0] * im_height),
				int(boxes[0,i,1]*im_width),
				int(boxes[0,i,2] * im_height),
				int(boxes[0,i,3]*im_width))
			l=[]
			boxes, scores, classes, num=boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])
			for i in range(len(boxes)):
				# Class 1 represents human in MS COCO
				if classes[i] == 1 and scores[i] > 0.3:		#set score value to whatever confidence value you want, depending on your model and it's accuracy
					l.append(boxes[i])
			print('new boxes processed')
			q.put(l)
		elif not x.empty() and not q.empty():
			print('clogging') 					#Hardcoded to not happen, but if it happens by any unpredicted error, will show clogging of communication queues between processes
