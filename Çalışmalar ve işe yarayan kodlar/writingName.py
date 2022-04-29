import os
# Function to rename multiple files
def main():
   i = 1
   path = "data_ders/data_images/"
   dosya = open("train.txt","w",encoding="utf-8")
#data_ders/data_images/1.jpg
   for filename in os.listdir(path):
      my_source = path + filename
      print(my_source)
      dosya.write(my_source+"\n")
      #my_dest = path + my_dest
      # rename() function will
      # rename all the files
      #os.rename(my_source, my_dest)
      i += 1
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()