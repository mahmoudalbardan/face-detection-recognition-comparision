import argparse
import os
import glob
import cv2
import align.detect_face as detect_face
import numpy as np
import tensorflow as tf
import datetime

from age_gender.wide_resnet import WideResNet
from time import time,sleep
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir, check_if_done, get_age_gender
from project_root_dir import project_dir
from src.sort import Sort

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = Logger()


def main():
    global colours, img_size
    args = parse_args()
    videos_dir = args.videos_dir
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold
    
    now = datetime.datetime.now()
    age,gender = None,None # age and gender init
    
    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:
            t = 64 # size of the image
            agender_model = WideResNet(t, depth=16, k=8)()
            agender_model.load_weights(args.agender_weights) # pretrained weights for age and gender recognition
    
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))

            minsize = 75
            # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709 # scale factor
            video_name = None
            while True:
#                year = str(now.year)
#                month = str(now.month)
#                day = str(now.day)
#                videos_dir = os.path.join(videos_dir,year,month,day)
                n_videos = len(glob.glob1(videos_dir,"*.mp4"))
                if n_videos>0:
                    list_of_videos = [video for video in os.listdir(videos_dir) if video.endswith("mp4")]
                    filename = list_of_videos[0]
                    video_name =os.path.join(videos_dir, filename)
                    if not filename.startswith("CAM_IN"):
                        print ("no corresponding videos are detected ... detetion")
                        os.remove(os.path.join(videos_dir, filename))
                        n_videos = n_videos - 1
                    if filename.startswith("CAM_IN") and check_if_done(video_name):
                        secs = 20
                        print ("FILE TRANSFER ALMOST DONE ... SLEEP FOR {} SECONDS".format(secs))
                        sleep(secs) # to wait for the transfer to be done
                        print ("WAKE UP AND READ THE VIDEO")
                        logger.info('Video_name:{}'.format(video_name))
                        cam = cv2.VideoCapture(video_name)
                        c = 0
                        k=n=1
                        while True:
                            ret, frame = cam.read()
                            if not ret:
                                logger.warning("ret false")
                                break
                            if frame is None:
                                logger.warning("frame drop")
                                break                                
                            if k>0 and k%n==0:
                                final_faces = []
                                addtional_attribute_list = []
                                #frame = rotate_bound(frame,90)
                                frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                                r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                if c % detect_interval == 0:
                                    img_size = np.asarray(frame.shape)[0:2]
                                    mtcnn_starttime = time()
                                    faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                                                                      factor)

                                    logger.info("MTCNN detect face cost time : {} s".format(
                                        round(time() - mtcnn_starttime, 3)))  # mtcnn detect ,slow
                                    face_sums = faces.shape[0]
                                    if face_sums > 0:
                                        face_list,genders,ages = [], [], []
                                        for i, item in enumerate(faces):
                                            score = round(faces[i, 4], 6)
                                            if score > face_score_threshold:
                                                det = np.squeeze(faces[i, 0:4])

                                                # face rectangle
                                                det[0] = np.maximum(det[0] - margin, 0)
                                                det[1] = np.maximum(det[1] - margin, 0)
                                                det[2] = np.minimum(det[2] + margin, img_size[1])
                                                det[3] = np.minimum(det[3] + margin, img_size[0])

                                                face_list.append(item)

                                                # face cropped
                                                bb = np.array(det, dtype=np.int32) # for face detection

                                                # use 5 face landmarks  to judge the face is front or side
                                                squeeze_points = np.squeeze(points[:, i])
                                                tolist = squeeze_points.tolist()
                                                facial_landmarks = []
                                                for j in range(5):
                                                    item = [tolist[j], tolist[(j + 5)]]
                                                    facial_landmarks.append(item)
                                                if args.face_landmarks:
                                                    for (x, y) in facial_landmarks:
                                                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                                                
                                                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()
                                                
                                                # age and gender estimation 
                                                gender,age = get_age_gender(agender_model,frame,bb)
                                                
                                                assert type(gender) ==  str
                                                assert type(age) == int
                                                
                                                # add gender and age to the list of attributes 
                                                genders.append(gender)
                                                ages.append(age)

                                                dist_rate, high_ratio_variance, width_rate = judge_side_face(np.array(facial_landmarks))

                                                # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face)
                                                item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate,age,gender]
                                                addtional_attribute_list.append(item_list)

                                        final_faces = np.array(face_list)
                            
                                trackers = tracker.update(final_faces, img_size, output_path,
                                                          addtional_attribute_list, detect_interval,age,gender)

                            k += 1
                            c += 1

                            for d in trackers:
                                if not no_display:
                                    d = d.astype(np.int32)
                                    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                                    if final_faces != []:
                                        cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.75,
                                                    colours[d[4] % 32, :] * 255, 2)
                                        cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                                    (1, 1, 1), 2)
                                    else:
                                        cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.75,
                                                    colours[d[4] % 32, :] * 255, 2)

                            if not no_display:
                                frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                                cv2.imshow("Frame", frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

                        # remove video
                        if os.path.exists(video_name):
                            #os.remove(video_name)
                            n_videos = n_videos - 1
                            print (" video removed from directory of camera 1 ...")
                
                if n_videos ==0:
                    print (" empty directory... waiting for new entries for camera 1")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=str,
                        help='Path to the data directory containing aligned your face patches.',
                        default='./videos_cam1')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='cam1')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    parser.add_argument('--agender_weights',
                        help='pretrained model for age and gender detection',
                        default="./age_gender/pretrained_models/weights.28-3.73.hdf5")
    args = parser.parse_args()
    return args




if __name__ == '__main__':  
    main()
    logger.info("---- End of detection phase camera 1-----")


