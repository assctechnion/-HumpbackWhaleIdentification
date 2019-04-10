import pandas as pd
import numpy as np
import shutil
import os
import sys

class data_set:
    def __init__(self,dir):
        self.dir = dir
        self.labels = pd.read_csv(dir + 'train.csv')
        self.value_counts = self.labels['Id'].value_counts()
        
    def split(self,val_ratio,class_sizelim = 0):

        train_dir = self.dir + 'train/'
        val_sep_dir = self.dir + 'val_sep/'
        train_sep_dir = self.dir + 'train_sep/'
        value_counts_for_val = np.ceil((self.value_counts * val_ratio))

        if not os.path.exists(train_sep_dir):
            os.mkdir(train_sep_dir)

        if not os.path.exists(val_sep_dir):
            os.mkdir(val_sep_dir)
            
        for filename, class_name in self.labels.values:
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
 
            if self.value_counts[class_name] < class_sizelim :
                continue
            if value_counts_for_val[class_name] > 0:
                value_counts_for_val[class_name] -= 1
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
     
    def inflate(self , upto):
         for folder in os.listdir(self.dir + 'train_sep/'):
            folder_path = self.dir + 'train_sep/' + folder
            class_list = os.listdir(folder_path)
            i = 0
            j = 0 
            while len(class_list) + j < upto:
                path = folder_path + '/' +  class_list[i]
                shutil.copy(path , path[:-4] + '_' + str(j) + '.jpg')
                j += 1
                i += 1
                if i == len(class_list):
                    i = 0
                    
                