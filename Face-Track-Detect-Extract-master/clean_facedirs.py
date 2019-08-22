from time import time
import os
from lib.utils import Logger
logger  = Logger()


def delete_images(in_path,out_path,max_allowed_time):
    """ delete all faces not recognized if the stay in the directory
        for max_allowed_time 
    """
    while True:
        image_names,paths = [],[]
        for img in os.listdir(in_path):
            if img.endswith(".jpg"):
                paths.append(os.path.join(in_path,img))
                image_names.append(img)
        for img in os.listdir(out_path):
            if img.endswith(".jpg"):
                paths.append(os.path.join(out_path,img))
                image_names.append(img)
                
          
        for i,name in enumerate(image_names):
            if os.path.exists(paths[i]):
                if os.stat(paths[i]).st_size/1000 > 15:
                    os.remove(paths[i])
                    
                name_psplit = name.split('.')[0]
                kept_since = int(name_psplit.split('_')[1])
                if int(time()) - kept_since > max_allowed_time:
                    os.remove(paths[i])
                else:
                    print ("No face image exceeded the maximum allowed time of {} seconds".format(int(max_allowed_time)))
        
  
    
if __name__ == '__main__':
    in_path ="./cam1"
    out_path ="./cam2"
    model_path ="./facenet/model"
    delete_images(in_path,out_path,7200)
    
    
    

