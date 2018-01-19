import sys
sys.path.append('../python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy

flow_frames = 'flow_images/'
RGB_frames = '/storage/sda1/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/crop_images_resize/'
test_frames = 5 
train_frames = 5
test_buffer = 1
train_buffer = 120

def processImageCrop(im_info, transformer, flow):
  im_path = im_info[0]
  #print im_path
  im_crop = im_info[1] 
  im_reshape = im_info[2]
  im_flip = im_info[3]
  data_in = caffe.io.load_image(im_path)
  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, im_reshape)
  if im_flip:
    data_in = caffe.io.flip_image(data_in, 1, flow) 
    data_in = data_in[im_crop[0]:im_crop[2], im_crop[1]:im_crop[3], :] 
  processed_image = transformer.preprocess('data_in',data_in)
  return processed_image
  
  
class ImageProcessorCrop(object):
  def __init__(self, transformer, flow):
    self.transformer = transformer
    self.flow = flow
  def __call__(self, im_info):
    return processImageCrop(im_info, self.transformer, self.flow)

class sequenceGeneratorSample(object):
  def __init__(self, buffer_size, clip_length, num_samples, sample_dict, sample_order):
    self.buffer_size = buffer_size   #24
    self.clip_length = clip_length   #16
    self.N = self.buffer_size*self.clip_length   #16*24
    self.num_samples = num_samples     #101
    self.sample_dict = sample_dict
    #{'vedioname':{'frames':'picturename','reshape':[320,240],'crop':[227,227],'num_frames':picturenumber,'label':pictureclass},...}
    self.sample_order = sample_order  #all video names in order
    self.idx = 0                    #picture index

  def __call__(self):
    label_r = []
    im_paths = []
    im_crop = []
    im_reshape = []  
    im_flip = []
 
    if self.idx + self.buffer_size >= self.num_samples:
      idx_list = range(self.idx, self.num_samples)
      idx_list.extend(range(0, self.buffer_size-(self.num_samples-self.idx)))     #(idx,num_videos,0,buffer_size+idx-num_videos)
    else:
      idx_list = range(self.idx, self.idx+self.buffer_size)   #(idx,idx+buffer)
    

    for i in idx_list:
      key = self.sample_order[i]
      sample_reshape = self.sample_dict[key]['reshape']
      sample_crop = self.sample_dict[key]['crop']
      for f in self.sample_dict[key]['frames']:    #crop_images/0851/0851_20130925_3M_relapse.jpg	0
        label_r.append(f.split('\t')[1])
        im_paths.append(f.split('\t')[0])
      im_reshape.extend([(sample_reshape)]*self.clip_length)
      r0 = int(random.random()*(sample_reshape[0] - sample_crop[0]))
      r1 = int(random.random()*(sample_reshape[1] - sample_crop[1]))
      im_crop.extend([(r0, r1, r0+sample_crop[0], r1+sample_crop[1])]*self.clip_length)     
      f = random.randint(0,1)
      im_flip.extend([f]*self.clip_length)
	  
        
    
    im_info = zip(im_paths,im_crop, im_reshape, im_flip)

    self.idx += self.buffer_size
    if self.idx >= self.num_samples:
      self.idx = self.idx - self.num_samples

    return label_r, im_info
	
def advance_batch(result, sequence_generator, image_processor, pool):
  
    label_r, im_info = sequence_generator()
    tmp = image_processor(im_info[0])
    result['data'] = pool.map(image_processor, im_info)
    result['label'] = label_r
    cm = np.ones(len(label_r))
    cm[0::5] = 0
    result['clip_markers'] = cm


class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)

class sampleRead(caffe.Layer):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.sample_list = 'test.txt' 

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.sample_list, 'r')
    f_lines = f.readlines()
    f.close()

    sample_dict = {}
    current_line = 0
    self.sample_order = []
    for line in f_lines:
        line=line.strip('\n')    #0866_20140106_6M_relapse.jpg	1
        sample_name=line.split('_')[0]
        #print sample_name
        line_path=self.path_to_images+sample_name+'/'+line    #crop_images/0851/0851_20130925_3M_relapse.jpg	0
        if sample_name in sample_dict:
            sample_dict[sample_name]['frames'].append(line_path)
        else:
            sample_dict[sample_name]={}
            sample_dict[sample_name]['frames']=[line_path]
            sample_dict[sample_name]['reshape']=(120,160)
            sample_dict[sample_name]['crop']=(112,112)
            sample_dict[sample_name]['num_frames']=5
            self.sample_order.append(sample_name)

    self.sample_dict = sample_dict
    self.num_samples = len(sample_dict.keys())
    
    #for i in self.sample_order:
#        key=self.sample_order[i]
    #print len(self.sample_order)
        #print self.sample_dict[f]['frames']

    #set up data transformer
    shape = (self.N, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    if self.flow:
      image_mean = [128, 128, 128]
      self.transformer.set_is_flow('data_in', True)
    else:
      channel_mean=np.load('./mnist_train_lmdb_mean.npy')
      print channel_mean.shape
      self.transformer.set_is_flow('data_in', False)
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = train_buffer

    self.image_processor = ImageProcessorCrop(self.transformer, self.flow)
    self.sequence_generator = sequenceGeneratorSample(self.buffer_size, self.frames, self.num_samples, self.sample_dict, self.sample_order)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, self.pool)
    #print self.thread_result.keys()
    self.dispatch_worker()
    self.top_names = ['data', 'label','clip_markers']
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'data':
        shape = (self.N, self.channels, self.height, self.width)
      elif name == 'label':
        shape = (self.N,)
      elif name == 'clip_markers':
        shape = (self.N,)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
    new_result_data = [None]*len(self.thread_result['data']) 
    new_result_label = [None]*len(self.thread_result['label']) 
    new_result_cm = [None]*len(self.thread_result['clip_markers'])
    for i in range(self.frames):
      for ii in range(self.buffer_size):
        old_idx = ii*self.frames + i
        new_idx = i*self.buffer_size + ii
        new_result_data[new_idx] = self.thread_result['data'][old_idx]
        new_result_label[new_idx] = self.thread_result['label'][old_idx]
        new_result_cm[new_idx] = self.thread_result['clip_markers'][old_idx]

    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'data':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data[i] 
      elif name == 'label':
        top[top_index].data[...] = new_result_label
      elif name == 'clip_markers':
        top[top_index].data[...] = new_result_cm

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class SampleReadTrain_flow(sampleRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.flow = True
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = flow_frames 
    self.sample_list = 'ucf101_split1_trainVideos.txt' 

class sampleReadTest_flow(sampleRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = True
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = flow_frames 
    self.sample_list = 'ucf101_split1_testVideos.txt' 

class sampleReadTrain_RGB(sampleRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.flow = False
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 112
    self.width = 112
    self.path_to_images = RGB_frames 
    self.sample_list = '/storage/sda1/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/5-cross/train5.txt' 

class sampleReadTest_RGB(sampleRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 112
    self.width = 112
    self.path_to_images = RGB_frames 
    self.sample_list = '/storage/sda1/lisa-caffe-public-lstm_video_deploy/examples/LRCN_activity_recognition/5-cross/test5.txt' 
	
	
