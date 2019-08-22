import argparse
import os
import glob
import imutils
import cv2
import align.detect_face as detect_face
import numpy as np
import random
import tensorflow as tf
import shutil

from collections import Counter
from time import time
from PIL import Image
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from vgg.face_detection_VGG import VGG
from src.sort import Sort
from facenet.src.compare import facenet_compare,load_model
from operator import itemgetter

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = Logger()


def compare2(sess,in_path,out_path,model_path,image_size,thr=1):
    """ comparison of two faces using facenet """
    n_imgout =  len(glob.glob1(out_path,"*.jpg"))
    n_imgin = len(glob.glob1(in_path,"*.jpg"))
    paths,probas,image_names = [],[],[]
    for img in os.listdir(in_path):
        if img.endswith(".jpg"):
            paths.append(os.path.join(in_path,img))
            image_names.append(img)
    for img in os.listdir(out_path):
        if img.endswith(".jpg"):
            paths.append(os.path.join(out_path,img))
            image_names.append(img)
            
    distance_matrix = facenet_compare(sess,model_path,paths,image_size)[-n_imgout:][:,:-n_imgout]
    #np.save("./matrix.npy",distance_matrix)
    
    min_inds,out_inds = [], []
    for k,row in enumerate(distance_matrix):
        s = set(row)
        min_ = sorted(s)[0]
        if min_ <=thr:
            min_inds.append(np.where(row==min_)[0][0])
            probas = [(1/ele)/np.sum(1/ele) for ele in row]
            out_inds.append(k+n_imgin)
#        else:
#            if abs(min - sorted(s)[1]) >=0.1:
#                min_inds.append(np.where(row==min)[0][0])
#                probas = [(1/ele)/np.sum(1/ele) for ele in row]

    return distance_matrix,min_inds,image_names,n_imgout,n_imgin,paths,probas,out_inds

def save_time(min_inds,paths,n_imgout,probas,out_inds):
    """ get time from name """
    if min_inds:
        names_in,names_out = handle_input_repetition(min_inds,out_inds,paths)
        assert len(names_in) == len(names_out)
        for i in range(len(names_in)):
            base_namein,base_nameout = os.path.basename(names_in[i]),os.path.basename(names_out[i])
            names_in_psplit, names_out_psplit = base_namein.split(".")[0],base_nameout.split(".")[0]
            exit_time, entry_time = names_in_psplit.split("_")[1],names_out_psplit.split("_")[1]
            try:
                entry_gender = names_in_psplit.split("_")[2]
                gender = entry_gender
                entry_age = [int(names_in_psplit.split("_")[3])]         
            except:
                entry_gender = 'xxx'
                gender = entry_gender
                entry_age = []
                
            try:
                exit_gender = names_out_psplit.split("_")[2]
                gender = exit_gender
                exit_age = [int(names_out_psplit.split("_")[3])]              
            except:
                exit_gender = 'xxx'
                gender = entry_gender
                exit_age = []
            
            if (exit_age+entry_age):
                age = int(np.mean((exit_age+entry_age)))
            else:
                age=0
            stay_time = int(exit_time) - int(entry_time)
            subject = names_out_psplit.split("_")[0]
           
            stat_file = open("./stat_file","a+")
            stat_file.write("{},{},{},{},{},{} \n".format(
                    subject,gender,age,int(exit_time),int(entry_time),stay_time,
                                                       )
                                        )
               
 
def handle_input_repetition(min_inds,out_inds,paths):
    """
    handle the following case: if more than output face corresponds to an input face
    """      
    count_in = Counter(min_inds)
    names_out,names_in = [],[]
    for ele in count_in:
        indices =  [i for i, x in enumerate(min_inds) if x == ele]
        out_ele = itemgetter(*indices)(out_inds)
        if isinstance(out_ele,int):
            out_ele = [out_ele]
        paths_ele = itemgetter(*out_ele)(paths)
        if isinstance(paths_ele,str):
            paths_ele = [paths_ele]
            
        names_in.append(paths[ele])
        names_out.append(max(paths_ele, key=os.path.getctime))
    return names_in,names_out
    
    
               
def main(sess,in_path,out_path,model_path):
    """ main """
    image_size=160
    while True:
        n_imgout =  len(glob.glob1(out_path,"*.jpg"))
        n_imgin = len(glob.glob1(in_path,"*.jpg"))
        if n_imgin > 0:
            if n_imgout > 0:
                start = time()
                matrix,min_inds,images,n_imgout,n_imagin,paths,probas,out_inds = compare2(sess,in_path,out_path,model_path,image_size)
                print ("Comparison time is {}".format( int(time()-start)))
                save_time(min_inds,paths,n_imgout,probas,out_inds)
                if min_inds:
                    
                    paths_to_remove_in = itemgetter(*min_inds)(paths)
                    if isinstance(paths_to_remove_in,str):
                        paths_to_remove_in = [paths_to_remove_in]

                    for path in paths_to_remove_in:
                        if os.path.exists(path):
                            os.remove(path)
                            
                    paths_to_remove_out = itemgetter(*out_inds)(paths)
                    if isinstance(paths_to_remove_out,str):
                        paths_to_remove_out = [paths_to_remove_out]

                    for path in paths_to_remove_out:
                        if os.path.exists(path):
                            os.remove(path)

                
                n_imgout = n_imgout - 1
            if n_imgout ==0:
                print ("Exit directory is empty ... waiting for new people to exit the store")
        if n_imgin ==0:
            print ("Entry directory is empty ... waiting for new people to enter the store")
    return matrix


if __name__ == '__main__':
    in_path ="./cam1"
    out_path ="./cam2"
    model_path ="./facenet/model"
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model(model_path)
            main(sess,in_path,out_path,model_path)





















#
#def which_image(verification,images) :
#    """
#        choose which image corresponds to the person in the output image
#        verification: (True or False, np.array(np.float) shape(1,2), bool)
#        1 case:
#        eliminate all True verification and choose the highest diff in the Falses
#        2 case:
#        if no True is found, also choose the highest diff in the Falses
#        3 case:
#        if all of them are True, choose the minimal diff in the Trues
#        """
#
#    truefalse   = [item[0] for item in verification]
#    probas      = np.array(([item[1] for item in verification]))
#    if True not in truefalse:  # kella false
#        array = np.abs(np.diff(probas,axis=1))
#        idx = int(np.where(array == np.min(array))[0][0])
#        return images[idx]
#
#    else:  # iza fiya true
#        if False in truefalse: # iza m5allata true w false
#            idxs = [i for i,ele in enumerate(truefalse) if ele==True]
#            probas_ = probas[np.asarray(idxs)]
#            images_ = [images[j] for j in idxs]
#            array =  np.abs(np.diff(probas_,axis=1))
#            idx = int(np.where(array == np.max(array))[0][0])
#            return images_[idx]
#
#        else: # iza kella true
#            array = np.abs(np.diff(probas,axis=1))
#            idx = int(np.where(array == np.max(array))[0][0])
#            return images[idx]
#
#
#def compare(obj,in_path,out_path):
#    """ compare faces between input and output directories and save to stat_file"""
#    ext = ".jpg"
#    data = []
#    for out_image in os.listdir(out_path):
#        if len(glob.glob1(out_path,"*.jpg")) >=1:
#            if out_image.endswith(ext):
#                verification = []
#                input_images = []
#                for in_image in os.listdir(in_path):
#                    if len(glob.glob1(in_path,"*.jpg")) >=1:
#                        if in_image.endswith(ext):
#                            img1 = os.path.join(out_path, out_image)
#                            img2 = os.path.join(in_path, in_image)
#                            verification.append(obj.verifyFace(img1,img2))
#                            input_images.append(in_image)
#
#                in_image = which_image(verification,input_images)
#                out_psplit , in_psplit = out_image.split('.')[0], in_image.split('.')[0]
#                out_usplit , in_usplit = out_psplit.split('_'), in_psplit.split('_')
#
#                stay_time = int(out_usplit[0]) - int(in_usplit[0])
#                try:
#                    gender    = out_usplit[1]
#                    age       =  np.mean([int(out_usplit[2]),int(in_usplit[2])])
#                except:
#                    gender = "xxx"
#                    age = "yy"
#
#
#                # remove the two photos
#                # os.remove(os.path.join(in_path, in_image)) # remove input image
#                # os.remove(os.path.join(out_path, out_image)) # remove output image
#
#
#        data.append([stay_time,gender,age])
#
#        stat_file = open("/Users/malbardan/Downloads/Face-Track-Detect-Extract-master/stat_file","a")
#        stat_file.write("{},{},{} \n".format(stay_time,gender,age))

