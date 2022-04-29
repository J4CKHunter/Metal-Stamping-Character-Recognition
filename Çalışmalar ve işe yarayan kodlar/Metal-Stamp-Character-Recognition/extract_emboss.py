import cv2
import numpy as np
import imutils
import os
import time
from recognition.num.find_number import my_recognizer as num_recognizer
from recognition.alpha.find_number import my_recognizer as char_recognizer
os.system('color 0a')
#Yukarıda kütüphaneler eklendi my recognizer olarak num ve char recognizerları çağırdı.
def main():
	try:
		print('>> Loading Code')
		begin()
		input('Press Enter to continue...')
	except Exception as e:
		print(e)
		input('Press Enter to continue...')
#def main içerisinde exception kontrolü yapıyor eğer exception yaşanırsa sadece sorunu gösteriyor. Eğer yaşanmazsa loading code yazısını gösterip ""begin fonksiyonunu başlatıyor. sonr ayazı yazdırıyor.
def imshow(img):
	cv2.imshow('test', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def my_thresholding(image):
	#adding white padding to each individual cropped character
	shape=image.shape
	#resim boyutunu alıyor
	#https://www.tutorialkart.com/opencv/python/opencv-python-get-image-size/
	w=shape[1]
	#w adlı değişkene 1. indexi atıyor
	h=shape[0]
	#h adlı değişkene 0. indexi atıyor
	base_size=h+20,w+20,3
	#base size belirliyor
	#make a 3 channel image for base which is slightly larger than target img
	base=np.zeros(base_size,dtype=np.uint8)
	#0'lar matrisi yapıyor base size da 
	#https://tr.csstricks.net/8226578-numpy-zeros-in-python-numpy-ones-in-python-with-example
	cv2.rectangle(base,(0,0),(w+20,h+20),(255,255,255),30)#really thick white rectangle
	#Hesaplanan kareye çizim yapılır
	base[10:h+10,10:w+10]=image
	#boş basein içine matrisi aktarıyor
	#cv2.imshow('bordered', base)
	#cv2.waitKey(0)
	#====================================================================================
	return base

def my_num_ocr(image):
	#from recognition.num.find_number import my_recognizer as num_recognizer
	try:
		number = num_recognizer.find(image)
		#num_recognizer kullanılarak modele ulaşılıyor ve numara alınıyor
		#print(str(number))
		#os.system('pause')
		return number
	except Exception as e:
		print(e)

def my_alpha_ocr(image):
	#from recognition.num.find_number import my_recognizer as char_recognizer
	try:
		alpha = char_recognizer.find(image)
		#char_recognizer kullanılarak modele ulaşılıyor ve harf alınıyor
		#print(str(number))
		#os.system('pause')
		return alpha
	except Exception as e:
		print(e)

def begin():
	print('>> Reading image')
	img = cv2.imread('raw/mstl2.png')
	#klasörden dosya okuyor onu img nin atıyor.
	print('>> Resizing input image by width = 900px')
	img = imutils.resize(img, width=900)
	#fotoğrafı imutils kütüphanesini kullanarak genişliği 900 olarak ayarlıyor
	cv2.imwrite('temp_photos/1.input_resized.png', img)
	#fotoğarıfın boyutu ayarlandıktan sonra kaydediyor.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Renk paletini gri tonlu hale getiriyor
	blurred = cv2.GaussianBlur(gray, (11, 11), 2)
	#Gauss bulanıklaştırması uygulanıyor kullanım şekli aşağıdaki linkte var 
	#https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
	kernel = np.ones((25,25), np.uint8)
	#np.ones matris oluşturmaya yarar Tüm elemanları bir olan 25'e 25 bir array
	#https://erdincuzun.com/numpy/01-numpy-array-nesnesi/
	#uint8, 0..255 değerlerini temsil edebilen işaretsiz 8 bitlik bir tamsayıdır. int ise genellikle 32 bit işaretli bir tamsayıdır. dtype=int kullanarak dizi oluşturduğunuzda, o dizideki her öğe 4 bayt alır.
	img_dilation = cv2.dilate(blurred, kernel, iterations=1)
	#Bu operatör giriş olarak verilen görüntü üzerinde parametreler ile verilen alan içerisindeki sınırları genişletmektedir, bu genişletme sayesinde piksel gurupları büyür ve pikseller arası boşluklar küçülür
	#Blurlanmış fotoğrafa kernel matrisine göre 1 iterasyon yaparak genişletme uygular
	gray = img_dilation
	#dilation uygulanmış fotoğrafı gray adlı nesneye aktarır.
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
	#Numpy yardımıyla manuel yapılandırma elemanı oluşturmaya yarar çekirdeğin şeklini ve boyutunu girmek gerekiyor matris içini hep 1 dolduruyor
	#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
	#morphologyEx genel morfoloji yönetimi içerisine yazdığımız cv2.MORPH_TOPHAT ile ne istediğimiz seçiliyor. Giriş görüntüsü ile görüntünün Açılması arasındaki farktır.
	#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	#https://github.com/mesutpiskin/computer-vision-guide/blob/master/docs/10-morfolojik-goruntu-isleme.md
	#Bu operatör giriş olarak verilen görüntüden, opening (açınım) operatörü uygulanmış halini çıkarır.  anladığım akdarıyla görüntüden matrisi çıkartıyor
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
	#Sobel filtesi kenarları tespit etmek için kullanılır. 1. sırada foto 2. sırada image matrixdeki dışta kalan piksellerin nasıl handle edileceği ile alakalı. Sobel fonksiyonun 3. ve 4. parametresi dx ve dy’dir. Eğer dx’e 1, dy’ye 0 verirsek bu horizontal (yatay) kenarları algılayacak. dx’e 0, dy’ye 1 verirsekte vertical (dikey) kenarları algılayacak.
	#https://abdulsamet-ileri.medium.com/görüntü-filtrelerini-uygulama-ve-kenarları-algılama-21d42f194db4
	gradX = np.absolute(gradX)
	#gradx'in mutlak değeri alınmış
	#https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	#gradx'in minimum değerini minVal'a gradx'in maksiumum değerini maxVal'a atamış
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	#gradX'ten min valı çıkarmış bunları maxvalla minval arasındaki farka bölmüş. ve bunları 255 ile çarpmış ve bunu tekrar gradXe atamış
	gradX = gradX.astype("uint8")
	#gradx data tipini 0 through 255 decimal e çevirmiş
	print('>> Finding vertical scharr gradient')
	#yazı yazdıyıror consoleda
	cv2.imwrite('temp_photos/2.input_vertical_scharr_gradient.png', gradX)
	#Gradx fotoğrafını 2. foto olarak kaydediyor.
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	#morphologyEx genel morfoloji yönetimi içerisine yazdığımız cv2.MORPH_CLOSE ile ne istediğimiz seçiliyor. Görüntüye dilation operatörü uygulanır ve ardından Erosion operatörü uygulanır. 
	#https://github.com/mesutpiskin/computer-vision-guide/blob/master/docs/10-morfolojik-goruntu-isleme.md
	thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	#hedefMat = cv.threshold(kaynakMat,esikDegeri,maxDeger,cv.threshoidngTipi)
	#Kaynak olarak alınan görüntü üzerindeki piksel,esikDegeri olarak verilen değerden büyükse maksDeger olarak verilen parametre değerine atanır. (Threshbinary)
	#https://github.com/mesutpiskin/computer-vision-guide/blob/master/docs/10-morfolojik-goruntu-isleme.md
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
	#Numpy yardımıyla manuel yapılandırma elemanı oluşturmaya yarar çekirdeğin şeklini ve boyutunu girmek gerekiyor matris içini hep 1 dolduruyor
	#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	#morphologyEx genel morfoloji yönetimi içerisine yazdığımız cv2.MORPH_CLOSE ile ne istediğimiz seçiliyor. Görüntüye dilation operatörü uygulanır ve ardından Erosion operatörü uygulanır. 
	#https://github.com/mesutpiskin/computer-vision-guide/blob/master/docs/10-morfolojik-goruntu-isleme.md
	print('>> Generating Horizontal projection to find text')
	#yazı yazdıyıror consoleda
	cv2.imwrite('temp_photos/3.input_Horizontal_projection.png', thresh)
	#thresh fotoğrafını 3. foto olarak kaydediyor.
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#Konturlar aynı renk ve yoğunluğa sahip olan tüm kesintisiz noktaları sınır boyunca birleştiren bir eğri olarak basitçe açıklanabilir.FindContours yöntemiyle konturleri bulunan resim komple değişir orjinal halini bir daha kullanılamaz hale gelir. Bunun için resimi yazılımda yedeklemeniz gerekmektedir.
	#Eğer cv2.CHAIN_APPROX_NONE komutu kullanılsaydı bütün kontur bilgileri saklanacaktı. fakat her durumda bütün kontur bilgilerine ihtiyaç varmıdır ve nasıl kullanabilir.Bir örnekle açıklayalım düz bir çizgiden oluşan bir konturu çizdirmek istenirse bütün kontur bilgilerine ihtiyaç yoktur başlangıç ve bitiş noktalarının koordinatlarını bilmek çizdirmek için yeterlidir. Bu durumda ise cv2.CHAIN_APPROX_SIMPLE komutunu kullanmak yeterlidir. 
	cnts = imutils.grab_contours(cnts)
	#Tuple değerini almaya yarıyor
	#https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
	#finding the largest area rectangular contour
	areas = [cv2.contourArea(c) for c in cnts]
	#anladığım kadarıyla ctns içinde contourları tek tek konturların alanlarını alıyor.
	max_index = np.argmax(areas)
	#içindeki max değeri vereni tespit ediyor
	#Argmax, bir hedef fonksiyondan maksimum değeri veren argümanı bulan bir işlemdir.
	#Argmax, tahmin edilen en büyük olasılığa sahip sınıfı bulmak için en yaygın olarak makine öğreniminde kullanılır.
	#Uygulamada argmax() NumPy işlevi tercih edilse de, Argmax manuel olarak uygulanabilir.
	#https://machinelearningmastery.com/argmax-in-machine-learning/
	cnt=cnts[max_index]
	#cnt değişkeni olarak cntsnin içindeki max değeri veren seçiliyor
	x,y,w,h = cv2.boundingRect(cnt)
	#cv2.boundingRect() metodu ile kontur çerçeve noktalarını hesaplanır
	cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)
	#Hesaplanan kareye çizim yapılır
	cropped_img = img[y-10:y+h+10, x-10:x+w+10]
	#resim kırpma ayarları yapılır
	cropped_img = imutils.resize(cropped_img, height=100)
	#resim kırpılır
	print('>> Cropping Text Roi')
	#konsola yazdırılır
	cv2.imwrite('temp_photos/4.cropped.png', cropped_img)
	#4. foto olarak kaydedilir

	#================= Character segmentation ================================
	image = cropped_img.copy()
	#image = cv2.imread('raw/b.jpg')
	#kırpılmış resmi kopyalayıp image resminin içine atıyor
	dst = cv2.fastNlMeansDenoisingColored(image.copy(), None, 10, 10, 7, 15)
	#Gürültü azaltmaya yarıyor
	#https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Non-local_Means_Denoising_Algorithm_Noise_Reduction.php
	#https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga03aa4189fc3e31dafd638d90de335617
	print('>> Removing Noise from cropped Text Roi')
	#consola yazdırıyor
	cv2.imwrite('temp_photos/5.cropped_denoise.png', dst)
	#5. görsel olarak kaydediyor
	gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
	#Renk paletini gri tonlu hale getiriyor
	blurred = cv2.GaussianBlur(gray, (3, 7), 13)
	#Gauss bulanıklaştırması uygulanıyor kullanım şekli aşağıdaki linkte var 
	#https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
	edge = cv2.Canny(blurred,80,30)
	#Kenar tespiti yapıyor 
	#https://medium.com/operations-management-türkiye/canny-kenar-tespiti-edge-detection-ca5fd65a8227
	print('>> Finding canny edge for text segmentation')
	#konsola yazdırıyor
	cv2.imwrite('temp_photos/6.cropped_canny_edge.png', edge)
	#6. fotoğraf olarak kaydediyor
	edge = np.uint8(edge)
	#edge'i np.uint8 data tipine çeviriyor
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,500))
	#Numpy yardımıyla manuel yapılandırma elemanı oluşturmaya yarar çekirdeğin şeklini ve boyutunu girmek gerekiyor matris içini hep 1 dolduruyor
	#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	closing = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
	#morphologyEx genel morfoloji yönetimi içerisine yazdığımız cv2.MORPH_CLOSE ile ne istediğimiz seçiliyor. Görüntüye dilation operatörü uygulanır ve ardından Erosion operatörü uygulanır. 
	#https://github.com/mesutpiskin/computer-vision-guide/blob/master/docs/10-morfolojik-goruntu-isleme.md
	print('>> Generating vertical projection')
	#konsola yazdırıyor
	cv2.imwrite('temp_photos/7.cropped_vertical_projection.png', closing)
	#7. fotoğraf olarak kaydediyor.

	kernel = np.ones((11,9), np.uint8)
	#np.ones matris oluşturmaya yarar Tüm elemanları bir olan 11'e 9 bir array
	#https://erdincuzun.com/numpy/01-numpy-array-nesnesi/
	img_dilation = cv2.dilate(edge, kernel, iterations=1)
	#Bu operatör giriş olarak verilen görüntü üzerinde parametreler ile verilen alan içerisindeki sınırları genişletmektedir, bu genişletme sayesinde piksel gurupları büyür ve pikseller arası boşluklar küçülür
	#Blurlanmış fotoğrafa kernel matrisine göre 1 iterasyon yaparak genişletme uygular
	edge = img_dilation
	#img dilation u edge içine atar

	ctrs,_ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
	#Konturlar aynı renk ve yoğunluğa sahip olan tüm kesintisiz noktaları sınır boyunca birleştiren bir eğri olarak basitçe açıklanabilir.FindContours yöntemiyle konturleri bulunan resim komple değişir orjinal halini bir daha kullanılamaz hale gelir. Bunun için resimi yazılımda yedeklemeniz gerekmektedir.
	#Eğer cv2.CHAIN_APPROX_NONE komutu kullanılsaydı bütün kontur bilgileri saklanacaktı. fakat her durumda bütün kontur bilgilerine ihtiyaç varmıdır ve nasıl kullanabilir.Bir örnekle açıklayalım düz bir çizgiden oluşan bir konturu çizdirmek istenirse bütün kontur bilgilerine ihtiyaç yoktur başlangıç ve bitiş noktalarının koordinatlarını bilmek çizdirmek için yeterlidir. Bu durumda ise cv2.CHAIN_APPROX_SIMPLE komutunu kullanmak yeterlidir. 
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	#cv2.boundingRect() metodu ile kontur çerçeve noktalarını hesaplanır
	#galiba ctrs içindeki ctrlerin çerçeve noktalarını hesaplayıp sıralamış
	idx = 1
	#idx' 1 değerini atamış
	segments = {}
	#segments adında boş liste oluşturmuş
	for i, ctr in enumerate(sorted_ctrs): 
		#enumerate() methodu itere edilebilir bir objenin(list, string, tuple vb) itemlarına birer index numarası verir.
		#i ve indexi ctr ise sorted ctrs deki değerleri tutuyor gibi anladım
	    x, y, w, h = cv2.boundingRect(ctr) 
	    #cv2.boundingRect() metodu ile kontur çerçeve noktalarını hesaplanır
	    roi = image[y:y+h, x:x+w] #bunu kullanmamış
	    # show ROI 
	    #cv2.imshow('segment no:'+str(i),roi) 
	    #cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 
	    #cv2.waitKey(0) 
	    if h>20 and w>10:
			#Eğer h 20 den büyükse w 10 dan büyükse
	    	cv2.rectangle(image,(x-7,y+10),( x + w, y + h-10 ),(0,255,),2)
			#Hesaplanan kareye çizim yapılır
	    	segments.update({idx:(x-7,y+10,x+w,y+h-10)})
			#segments listesine ekleme yapar değerleri idx galiba index oluyor eklenen değerlerin
	    	idx+=1 
			#idx i bir arttırır
	print('>> Segmenting individual Character')
	#konsola yazdırır
	cv2.imwrite('temp_photos/8.segmented_text.png', image)
	#8. foto olarak kaydeder
	cv2.imshow('segmented text',image)
	#8. fotoyu gösterir

	#==============Now we will crop out each character segment for recognition ========
	#print(segments)

	text_segment = {}
	#text segment adında liste oluşturuyor
	for i in range(1,len(segments)+1):
		#for döngüsü oluşturuyor i sayacı kullanıyor 1 den segment listesinin uzunluğunun 1 fazlasına
		x,y,w,h = segments[i]
		#segment listesindeki değerleri ayrı ayrı değişkenlere aktarıyor
		cropped_segment = cropped_img[y:h, x:w]
		#anlamadım
		#======================= Now applying thresholding on each character ==========
		cropped_segment = my_thresholding(cropped_segment)
		#Cropped segment resmini my thresholding fonksiyonuna gönderiyor gelen cevabı cropped segment e aktarıyor
		#==============================================================================
		text_segment.update({i:cropped_segment})
		#text segment listesini i indexiyle ve cropped segment değişkeniye sürekl güncelliyor
		cv2.imwrite('temp_photos/segments/'+str(i)+'.png',cropped_segment)
		#fotoğrafların her birini tek tek kaydediyor

	#imshow(cropped_segment)
	idx = 1
	#index'e 1 atıyor
	final_ocr = ''
	#final ocr adında değişken tanımlamış
	for i in range(1,len(segments)+1):
		#for döngüsü oluşturuyor i sayacı kullanıyor 1 den segment listesinin uzunluğunun 1 fazlasına
		if idx == 1 or idx == 4 or idx == 5:#false
			#Eğer index 1 yada index 4 yada index 5 ise
			#block for alphabet recognition
			
			any_char = text_segment[i] # filtering alpha characters as per position
			#Text segmentin i indexini anychar'a ata
			#recognizing a alphabet
			alpha_char = my_alpha_ocr(any_char)
			#any_char'ı my_alpha_ocr'a gönderiyor sonucu alpha_char'a aktarıyor
			#print('position: '+str(i)+', Recognized: '+str(alpha_char))
			final_ocr = str(final_ocr+str(alpha_char))
			#final_ocr yani en son ki doğruluk oranı alpha_char dan gelen harfle toplanıyor bu sayede okunan kelimeler listesi elde ediliyor
			#print('position: '+str(i)+', Recognized: X')
			#final_ocr = str(final_ocr+'X') #filing with x when not recognizing
			#cv2.imshow('any char', any_char)
			#cv2.waitKey(0)
			idx+=1
			#index 1 arttırılıyor
		else:
			#block for number recognition
			#eğer index 1,4,5 değilse burası çalışıyor
			any_char = text_segment[i] # filtering number characters as per position
			#Text segmentin i indexini anychar'a ata
			#recognizing a number
			num_char = my_num_ocr(any_char)
			#any_char'ı my_num_ocr'a gönderiyor sonucu num_char'a aktarıyor
			#print('position: '+str(i)+', Recognized: '+str(num_char))
			final_ocr = str(final_ocr+str(num_char))
			#final_ocr yani en son ki doğruluk oranı alpha_char dan gelen harfle toplanıyor bu sayede okunan kelimeler listesi elde ediliyor
			#cv2.imshow('any char', any_char)
			#cv2.waitKey(0)
			idx+=1
			#index 1 arttırılıyor
	print()
	print('OCR: '+final_ocr)
	#en son hepsini yazdırıyor
	cv2.waitKey(0)

#==============================
if __name__ == '__main__':
	main()