import cv2
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import glob

def normalize(arr):
    arr = arr.astype('float32')
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


# Dimensions of test images
row_length = 480 * 270


#Reads images in train/test set
def read_images(folder):
    images = []
    mat = np.zeros((1, row_length), np.float32)
    folder = folder + "*/*/*/*"
    for filename in sorted(glob.glob(folder)):
        img2 = cv2.imread(os.path.join(folder, filename), 1)
        img = cv2.imread(os.path.join(folder, filename), 0)

        width, height = img.shape[:2]
        if img is not None:
            images.append(img2)
            outimg = np.reshape(img, (1, width * height))
            mat = np.vstack((mat, outimg))

    mat = np.delete(mat, 0, axis=0)
    return images, mat




#Read images one person at a time
def read_images_new(folder):
    images = []
    mat = np.zeros((1, row_length), np.float32)
    folder = folder + "*"
    for filename in sorted(glob.glob(folder)):
        img2 = cv2.imread(os.path.join(folder, filename), 1)
        img = cv2.imread(os.path.join(folder, filename), 0)
        width, height = img.shape[:2]

    mat = np.delete(mat, 0, axis=0)
    return images, mat



#Read answers multiple annotations at a time
def read_answers(filename):
    returnlist = []
    with open(filename) as f:
        content = f.readlines()
        values = [x.split('\t') for x in content]
        for y in range(len(values)):
            num = int(values[y][1])
            for x in range(num):
                returnlist.append(np.float32(values[y][0]))
        return np.asarray(returnlist)

#Read answers one line at a time
def read_answers_old(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip('\n') for x in content]
        return np.asarray(content)


#Read in the frames of one video
def read_video(filename, colorFlag=0):
    cap = cv2.VideoCapture(filename)
    outputlist = []
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        if colorFlag == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.pyrDown(frame)
        frame = cv2.pyrDown(frame)
        outputlist.append(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    return np.asarray(outputlist)


#Read all the videos in a folder
def read_all_video(folder, colorFlag=0):
    output = []
    folder = os.getcwd() + folder
    folder = folder + "*"
    for filename in sorted(glob.glob(folder)):
        print filename
        output.append(read_video(filename))
    return (output)


#Get an array of all the frames in a video
def get_all_frames(folder, colorFlag=0):
    frames = []
    video = read_all_video(folder, colorFlag)
    for vid in video:
        length, width, height = vid.shape

        for x in range(length):
            frames.append(vid[x])

    return frames


#Return a clip of a video from start to end seconds
def clipvideo(vid, start, end, fps, flow=0):
    retclip = []
    start *= fps
    end *= fps
    if flow == 0:
        end += 1
    for x in range(start, end):
        retclip.append(vid[x])
    return np.asarray(retclip)


#Return the optical flow frames concatenated in blocks of size blockSize x blockSize
def getFlowVid(vid,blockSize):
    retval = []

    length, width, height = vid.shape
    winwid = width/blockSize
    winhei = height/blockSize
    for x in range(length-1):
        flow = cv2.calcOpticalFlowFarneback(vid[x], vid[x+1], None, 0.5, 2, 32*32, 3, 5, 1.2, 0)
        xflow = cv2.resize(flow[:,:,0], (blockSize, blockSize))
        yflow = cv2.resize(flow[:,:,1], (blockSize, blockSize))
        retval.append(xflow)
        retval.append(yflow)
    return np.asarray(retval)


#Rearrange code for input to network
def getChannelsinVid(vid, colorFlags=0):
    retlist = []
    if colorFlags == 0:
        for x in range(1):
            retlist.append(vid)
    else:
        for x in range(3):
            retlist.append(vid)

    return retlist


#Returns video segments of desired size for an entire video
def makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0,  flow = 0):

    if colorFlag == 0:
        numFrames, wid, hei= video.shape
    else:
        numFrames, wid, hei, channels = video.shape
    normRate = normalFrameRate / desiredFrameRate
    secondsLong *= desiredFrameRate
    startsEverySecond *= desiredFrameRate
    returnList = []
    willtheRealReturnListPleaseStandUp = []
    for x in range(numFrames):
        if x % normRate == 0:
            returnList.append(video[x])
    if flow == 0:
        for x in range(len(returnList)):
            if (x % startsEverySecond == 0) and ((x+secondsLong + 1) < len(returnList)):
                clip = clipvideo(returnList, (x / desiredFrameRate), (x+secondsLong)/desiredFrameRate, desiredFrameRate, flow=flow)
                willtheRealReturnListPleaseStandUp.append(clip)
    else:
        for x in range(len(returnList)):
            if (x % startsEverySecond == 0) and ((x + secondsLong) < len(returnList)):
                clip = clipvideo(returnList, (x / desiredFrameRate), (x + secondsLong) / desiredFrameRate,
                                 desiredFrameRate, flow=flow)
                willtheRealReturnListPleaseStandUp.append(clip)

    print len(willtheRealReturnListPleaseStandUp)
    return willtheRealReturnListPleaseStandUp



#Makes the video segments for all the videos in a folder
def collectVidSegments(folder, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond, colorFlag=0, flow = 0):
    folder = folder + "*"
    returnList = []
    for filename in sorted(glob.glob(folder)):
        video = read_video(filename, colorFlag)

        lst = makeVidSegments(video, normalFrameRate, desiredFrameRate, secondsLong, startsEverySecond,colorFlag=colorFlag, flow=flow)

        for seg in lst:
            returnList.append(seg)

    return returnList

