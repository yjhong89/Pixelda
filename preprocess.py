import glob
import os
import scipy.misc
import numpy as np
import sys

source_dir = 'data_source'

dir_list = os.listdir(source_dir)
for d in dir_list:
  filenames = glob.glob(os.path.join(source_dir,d,'*.png'))
  print(d)
  for filename in filenames:
    f_name = filename.split('/')[2]
    if f_name[-8:] == 'mask.png' and f_name[:4]!= 'gray':
      im = scipy.misc.imread(os.path.join(source_dir,d,f_name))
      im_mask = (im != 178)
      gray_mask = np.round(np.dot(im_mask[...,:3],[0.299,0.587,0.114])) * 255
      gray_mask = np.expand_dims(gray_mask,2)
      rgb_f_name = os.path.join(source_dir,d,f_name[:-9]+'.png')
      rgb_im = scipy.misc.imread(rgb_f_name)
      merged_im = np.concatenate((rgb_im,gray_mask),2)
      scipy.misc.imsave(os.path.join(source_dir,d,'merged'+f_name[:-9]+'.png'),merged_im)

