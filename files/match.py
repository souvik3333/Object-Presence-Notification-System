from PIL import Image
from PIL import ImageChops
import numpy as np
import cv2
import numpy as np

def mean_square_error(path1,path2):
    image1 = cv2.imread(path1) 
    image2 = cv2.imread(path2) 
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("tmp/gray1.jpg", gray1)
    cv2.imwrite("tmp/gray2.jpg", gray2)
    def crop(image_path, coords, saved_location):
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)
    
    x1=gray1.shape[1]
    y1 = gray1.shape[0]
    x2=gray2.shape[1]
    y2 = gray2.shape[0]
    x=min(x1,x2)
    y=min(y1,y2)
    x1min=np.floor((x1-x)/2)
    y1min=np.floor((y1-y)/2)
    x1max=np.floor((x1+x)/2)
    y1max=np.floor((y1+y)/2)

    x2min=np.floor((x2-x)/2)
    y2min=np.floor((y2-y)/2)
    x2max=np.floor((x2+x)/2)
    y2max=np.floor((y2+y)/2)
    crop("tmp/gray1.jpg",(x1min,y1min,x1max,y1max),"tmp/gray1.jpg")
    crop("tmp/gray2.jpg",(x2min,y2min,x2max,y2max),"tmp/gray2.jpg")
    image1 = cv2.imread('tmp/gray1.jpg') 
    image2 = cv2.imread('tmp/gray2.jpg') 
    # print(image1.shape)
    # print(image2.shape)
    def MSE(img1, img2):
        squared_diff = (img1 -img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
        err = summed / num_pix
        return err
    return MSE(image1,image2)
    # return mean_square_error(,"2.jpg")