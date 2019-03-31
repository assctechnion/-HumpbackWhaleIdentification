from __future__ import print_function
import pandas as pd
import numpy as np
import shutil
import os
import sys

labels = pd.read_csv('train.csv')

# Create `train_sep` directory
train_dir = 'train/'
val_sep_dir = 'val_sep/'
train_sep_dir = 'train_sep/'
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

if not os.path.exists(val_sep_dir):
    os.mkdir(val_sep_dir)
    
valuecnts = labels['Id'].value_counts()
prt = 0.25
valuecnts_for_val = np.ceil((valuecnts * prt))

#data class samples threshold
clslim = 5

for filename, class_name in labels.values:
    
	# copying 1 sample classes both to train and val
	if valuecnts[class_name] == 1:
		# copying to train
		# Create subdirectory with `class_name`
        if not os.path.exists(val_sep_dir + class_name):
            os.mkdir(val_sep_dir + class_name)
        src_path = train_dir + filename 
        dst_path = val_sep_dir + class_name + '/' + filename 
        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                  .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                  .format(src_path, dst_path, sys.exc_info()))
		# copying to val
		# Create subdirectory with `class_name`
        if not os.path.exists(train_sep_dir + class_name):
            os.mkdir(train_sep_dir + class_name)
        src_path = train_dir + filename 
        dst_path = train_sep_dir + class_name + '/' + filename 
        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                  .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                  .format(src_path, dst_path, sys.exc_info()))
				  
	if valuecnts[class_name] < clslim :
        continue
    if valuecnts_for_val[class_name] > 0:
        valuecnts_for_val[class_name] -= 1
        # Create subdirectory with `class_name`
        if not os.path.exists(val_sep_dir + class_name):
            os.mkdir(val_sep_dir + class_name)
        src_path = train_dir + filename 
        dst_path = val_sep_dir + class_name + '/' + filename 
        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                  .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                  .format(src_path, dst_path, sys.exc_info()))
    else:
        # Create subdirectory with `class_name`
        if not os.path.exists(train_sep_dir + class_name):
            os.mkdir(train_sep_dir + class_name)
        src_path = train_dir + filename 
        dst_path = train_sep_dir + class_name + '/' + filename 
        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                  .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                  .format(src_path, dst_path, sys.exc_info()))