import cv2
import imgdet
import time
from imutils.video import VideoStream
import imutils
import multiprocessing
if __name__ == "__main__":
	model_path = r'C:\Users\Lenovo\Dropbox\My PC (LAPTOP-1OHMGAJG)\Desktop\PDC_Proj\frozen_inference_graph.pb'		#Change address to location of model's frozen_inference_graph.pb file
	q=multiprocessing.Queue()												#Queue to recieve bounding boxes from predictor
	x=multiprocessing.Queue()												#Queue to send image to predictor
	p1=multiprocessing.Process(target=imgdet.processFrame,args=(model_path,x,q))						#defining subprocess p1 as imgdet
	p1.daemon=True														#Set daemon process so as to terminate it when the main program terminates
	p1.start()														
	#usingPiCamera = False
	#frameSize = (1280, 720)
	#vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize,						# Initialize mutithreading the video stream.( NOT FOR NVIDIA JETSON)
	#		framerate=60).start()  
	print('new process started')
	time.sleep(10)														#Give time to p1 to finish up setting modules
	print('processes resumes')
	vs=cv2.VideoCapture(0,cv2.CAP_DSHOW)													# Using webcam
	c=0															#initialize counter to prevent predictor getting clogged
	fno,box=0,[(0,0,0,0)]
	g=time.time()
	while True:
		start=time.time()
		r,img = vs.read()
		img=cv2.resize(img,(1280,720))											#Resizing image to 720P
		#blank_image = np.zeros((720,1280,3), np.uint8)
		#img=img+blank_image
		fno+=1
		#If Q is empty and predictor is idle, send current frame to predictor via X Queue, c=1 means predictor busy
		if q.empty() and c==0:
			x.put(img)
			print('"Q" Queue empty, Added image to "X" queue')
			c=1
		#If Q has bounding boxes, recieve and update c to show that predictor is idle now		
		if not q.empty():
			print('recieved boxes from "Q" queue')
			box=q.get()
			print('boxes updated')
			c=0
		for i in box:
			cv2.rectangle(img,(i[1],i[0]),(i[3],i[2]),(255,0,0),2)				
			

	
		cv2.imshow('Preview',img)
		#print('FPS of camera:',fno/(time.time()-g))
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
	
