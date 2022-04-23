import cv2
import numpy
from PIL import Image
import imutils
from pytesseract import pytesseract
from PIL import ImageFilter

imageAdress = "Dataset/2.jpg" # bu şekilde kalsın şimdi gürülti kaldırıcam
imageAdress2 = "Dataset/spotify2.jpg"

#C:\Program Files (x86)\Tesseract-OCR
pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def detectCropPhoto(imageAdress):
    img = Image.open(imageAdress)
    width, height = img.size
    img = cv2.imread(imageAdress)
    if width > 180 and height >60:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)
        kernel = np.ones((25,25), np.uint8)
        img_dilation = cv2.dilate(blurred, kernel, iterations=1)
        gray = img_dilation
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
        gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")
        print('Dikey scharr gradyanı bulma')
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        print('Metni bulmak için Yatay projeksiyon oluşturma')
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        cnt=cnts[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        return(x,y,w,h)
        
    else:
        img = cv2.imread(imageAdress)
        return img

def cropPhoto(x,y,w,h,img):
    cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)
    cropped_img = img[y-10:y+h+10, x-10:x+w+10]
    cropped_img = imutils.resize(cropped_img, height=100)
    print('resim kırpıldı')
    #konsola yazdırılır
    return cropped_img



mainImg = cv2.imread(imageAdress)
mask = cv2.imread(imageAdress)
#mask sayesinde okunabilecek hale getirmeliyiz

img = Image.open(imageAdress)
width, height = img.size
#filtreler gelicek buradan sonra

mask = img.filter(ImageFilter.EMBOSS)

mask.show()


ele = numpy.pi/2.2 # radians
azi = numpy.pi/4.  # radians
dep = 10.          # (0-100)

# get a B&W version of the image
img = Image.open('daisy.jpg').convert('L') 
# get an array
a = numpy.asarray(img).astype('float')
# find the gradient
grad = numpy.gradient(a)
# (it is two arrays: grad_x and grad_y)
grad_x, grad_y = grad
# getting the unit incident ray
gd = numpy.cos(ele) # length of projection of ray on ground plane
dx = gd*numpy.cos(azi)
dy = gd*numpy.sin(azi)
dz = numpy.sin(ele)
# adjusting the gradient by the "depth" factor
# (I think this is how GIMP defines it)
grad_x = grad_x*dep/100.
grad_y = grad_y*dep/100.
# finding the unit normal vectors for the image
leng = numpy.sqrt(grad_x**2 + grad_y**2 + 1.)
uni_x = grad_x/leng
uni_y = grad_y/leng
uni_z = 1./leng
# take the dot product
a2 = 255*(dx*uni_x + dy*uni_y + dz*uni_z)
# avoid overflow
a2 = a2.clip(0,255)
# you must convert back to uint8 /before/ converting to an image
img2 = Image.fromarray(a2.astype('uint8')) 






if width > 180 and height >60:
    x,y,w,h = detectCropPhoto(imageAdress)
    mainImg= cropPhoto(x,y,w,h,mainImg)
    mask= cropPhoto(x,y,w,h,mask)



#kernel = np.ones((5,5), np.uint8)
#img_erosion = cv2.erode(mainImg, kernel, iterations=1)
#img_dilation = cv2.dilate(mainImg, kernel, iterations=1)









mainImg= np.int32(mainImg)
mask= np.int32(mask)

lastImage = mainImg - mask

img = np.array(lastImage, dtype=np.uint8)


words_in_image =pytesseract.image_to_string(img)

print(words_in_image)




cv2.imshow('test', img)
cv2.waitKey(0)




