
import numpy as np
import glob
import sys
sys.path.insert(0,'../../python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import pickle

cross = ['1','2','3','4','5']
for i in range(5):
	RGB_video_path = '/home/shiyan/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/3-predict-4/'+cross[i]+'/'
	#RGB_video_path = 'crop_images/'
	SAMPLE_LABEL_PATH='/home/shiyan/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/3-predict-4/sample_label_three_'+cross[i]+'.txt'
	#IMAGE_LABEL_PATH='test_four.txt'
	RESULT_LABELS_PATH='./5-6model_three-'+cross[i]+'.txt'
	#RESULT_PROBS_PATH='liulin/p_four1.txt'
	SAMPLE_NUMBER=3

	lstm_model = 'deploy_lstm.prototxt'

	RGB_lstm = '../snapshots_alexnet_lstm_RGB_iter_2000.caffemodel'
	#if len(sys.argv) > 1:
	#  video = sys.argv[1]
	#else:
	#  video = '0001'

	#Initialize transformers

	def initialize_transformer(image_mean, is_flow):
	  shape = (10*SAMPLE_NUMBER, 3, 227, 227)
	  transformer = caffe.io.Transformer({'data': shape})
	  channel_mean = image_mean
	  transformer.set_mean('data', channel_mean)
	  transformer.set_raw_scale('data', 255)
	  transformer.set_channel_swap('data', (2, 1, 0))
	  transformer.set_transpose('data', (2, 0, 1))
	  transformer.set_is_flow('data', is_flow)
	  return transformer

	ucf_mean_RGB=np.load('/home/shiyan/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/alexnet_lstm/mnist_train_lmdb_mean.npy')

	transformer_RGB = initialize_transformer(ucf_mean_RGB, False)

	# Extract list of frames in video
	#RGB_frames = glob.glob('%s%s/*.jpg' %(RGB_video_path, video))

	#classify video with LRCN model
	def LRCN_classify_video(frames, net, transformer, is_flow):
	  clip_length = SAMPLE_NUMBER
	  #print len(clip_input)
	  #offset = 8
	  input_images = []
	  for im in frames:
	    input_im = caffe.io.load_image(im)
	    if (input_im.shape[0] < 240):
	      input_im = caffe.io.resize_image(input_im, (240,320))
	    input_images.append(input_im)
	  vid_length = len(input_images)
	  input_data = input_images
	  output_predictions = np.zeros((5,2))


	  clip_input = input_data[0:-1]
	  print len(clip_input)
	  clip_input = caffe.io.oversample(clip_input,[227,227])
	  #print clip_input.shape[0]
	  clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
	  clip_clip_markers[0:10,:,:,:] = 0
	#    if is_flow:  #need to negate the values when mirroring
	#      clip_input[5:,:,:,0] = 1 - clip_input[5:,:,:,0]
	  caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
	  #print caffe_in.shape
	  for ix, inputs in enumerate(clip_input):
	    caffe_in[ix] = transformer.preprocess('data',inputs)
	  out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
	  #print caffe_in.shape
	  #print clip_clip_markers.shape
	  #print out['probs'].shape
	  #print np.mean(out['probs'],1)
	  output_predictions[0:5] = np.mean(out['probs'],1)
	  print output_predictions
	  #print output_predictions[-1].argmax()
	  return output_predictions[SAMPLE_NUMBER-1].argmax(), output_predictions
	 

	#Models and weights

	#lstm_model = 'deploy_lstm.prototxt'

	#RGB_lstm = 'snapshots_lstm_RGB_3_iter_2000.caffemodel'

	RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

	f=open(SAMPLE_LABEL_PATH,'r')
	f_lines=f.readlines()
	f.close()
	samples=[]
	labels=[]
	count=0
	for s in f_lines:
	  s=s.strip('\n')
	  samples.append(s.split('\t')[0])
	  #label=int(s.split('\t')[1])
	  labels.append(int(s.split('\t')[1]))

	predict_labels=[]
	predictions_RGB_LRCNS=[]
	l0_p0 = []
	l0_p1 = []
	l1_p1 = []
	l1_p0 = []
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for video in samples:
	  print video
	  RGB_frames = glob.glob('%s%s/*.jpg' %(RGB_video_path, video))
	  class_RGB_LRCN, predictions_RGB_LRCN = \
		 LRCN_classify_video(RGB_frames, RGB_lstm_net, transformer_RGB, False)
	  predict_labels.append(class_RGB_LRCN)
	  predictions_RGB_LRCNS.append(predictions_RGB_LRCN)
	#print predictions_RGB_LRCNS[0]
	for i in range(len(samples)):
		if labels[i] == 0 and predict_labels[i] == 0:
			TN += 1
			count += 1
			l0_p0.append(samples[i]+'\tp_labels:'+str(predict_labels[i])+'\t'+str(predictions_RGB_LRCNS[i][SAMPLE_NUMBER-1]))
		if labels[i] == 0 and predict_labels[i] == 1:
			FP += 1
			l0_p1.append(samples[i] + '\tp_class:' + str(predict_labels[i]) + '\t' + str(predictions_RGB_LRCNS[i][SAMPLE_NUMBER - 1]))
		if labels[i] == 1 and predict_labels[i] == 1:
			TP += 1
			count += 1
			l1_p1.append(samples[i]+'\tp_labels:'+str(predict_labels[i])+'\t'+str(predictions_RGB_LRCNS[i][SAMPLE_NUMBER-1]))
		if labels[i] == 1 and predict_labels[i] == 0:
			FN += 1
			l1_p0.append(samples[i]+'\tp_labels:'+str(predict_labels[i])+'\t'+str(predictions_RGB_LRCNS[i][SAMPLE_NUMBER-1]))

	print 'TP:' + str(TP) + '   FN:' + str(FN) + '   TN:' + str(TN) + '   FP:' + str(FP)
	Accuracy = (TP + TN) * 1.0 / (TP + FN + TN + FP)
	Sensitivity = TP * 1.0 / (TP + FN)
	Specificity = TN * 1.0 / (TN + FP)

	#write test result
	file = open(RESULT_LABELS_PATH, 'w')
	file.writelines('real_label_0:\n')
	for i in l0_p0:
		file.writelines(i + '\n')
	for i in l0_p1:
		file.writelines(i + '\n')
	file.writelines('-------------------------------------------------------\n')
	file.writelines('real_label_1:\n')
	for i in l1_p1:
		file.writelines(i + '\n')
	for i in l1_p0:
		file.writelines(i + '\n')
	#file_object.close()
	#f.close()

	file.writelines('TP:' + str(TP) + '   FN:' + str(FN) + '   TN:' + str(TN) + '   FP:' + str(FP))
	file.writelines('\n')
	file.writelines('Accuracy:'+str(Accuracy)+'   Sensitivity:'+str(Sensitivity)+'   Specificity:'+str(Specificity))
	file.close()
	#print 'TP:' + str(TP) + '   FN:' + str(FN) + '   TN:' + str(TN) + '   FP:' + str(FP)
	print 'Accuracy:'+str(Accuracy)+'   Sensitivity:'+str(Sensitivity)+'   Specificity:'+str(Specificity)
	print count*1.0/len(samples)
	#print 'positive class number: '+str(sum(new_labels))
	#print 'negtive class number: '+str(len(new_labels)-sum(new_labels))
	#print 'TP number: '+str(p_count)
	#print 'TN number: '+str(n_count)

	#f.close()



	del RGB_lstm_net




	def compute_fusion(RGB_pred, flow_pred, p):
	  return np.argmax(p*np.mean(RGB_pred,0) + (1-p)*np.mean(flow_pred,0))  

	#Load activity label hash
	#action_hash = pickle.load(open('action_hash_rev.p','rb'))
	#f=open(SAMPLE_LABEL_PATH,'r')
	#f_lines=f.readlines()
	#f.close()
	#action_hash={}
	#for line in f_lines:
	#	line=line.strip('\n')
	#	name=line.split('\t')[0]
	#	label=line.split('\t')[1]
	#	if name not in action_hash:
	#		action_hash[name]=label


	#print "RGB single frame model classified video as: %s.\n" %(action_hash[class_RGB_singleFrame])
	#print "Flow single frame model classified video as: %s.\n" %(action_hash[class_flow_singleFrame])
	#print "RGB LRCN model classified video as: %s.\n" %(action_hash[class_RGB_LRCN])
	#print "Flow LRCN frame model classified video as: %s.\n" %(action_hash[class_flow_LRCN])
	#print "0.5/0.5 single frame fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_singleFrame, predictions_flow_singleFrame, 0.5)])
	#print "0.33/0.67 single frame fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_singleFrame, predictions_flow_singleFrame, 0.33)])
	#print "0.5/0.5 LRCN fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_LRCN, predictions_flow_LRCN, 0.5)])
	#print "0.33/0.67 LRCN fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_LRCN, predictions_flow_LRCN, 0.33)])






