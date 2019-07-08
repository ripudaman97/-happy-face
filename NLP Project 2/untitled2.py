
import cv2
import xlsxwriter
#sample image names

names=["download","download (1)","download (2)","download (3)","download (4)","download (5)","download (6)"
,"download (7)","download (8)","download (9)","download (10)","download (11)","download (12)","download (13)"
,"download (14)","download (15)","download (16)","download (17)","download (18)","download (19)"
,"download (20)","download (21)","download (22)"]

#haarcascade for smile

face_cascade=cv2.CascadeClassifier("abc.xml")

row=1
col=0

workbook = xlsxwriter.Workbook('demo5.xlsx')
worksheet = workbook.add_worksheet()
for i in names:
    
    img=cv2.imread (""+i+".jpg",1)
    faces=face_cascade.detectMultiScale(img,scaleFactor=1.05)
    sum1=1
    sum3=1
    i=1
    for (x,y,w,h) in faces:
        i=i+1
        print(w)
        sum1+=w
        sum3+=h
    sumh=sum3/i    
    sumw=sum1/i    
    sumr=sumw/sumh
    row+=1
    worksheet.write(row, col+2,sumh)
    worksheet.write(row, col+3,sumw)
    worksheet.write(row, col+4,sumr)    
    if i<16:
         worksheet.write(row, col+7,1) #first 17 pictures have a happy face
    else:
         worksheet.write(row, col+7,2) #next have a sad face(classifying manually to train model)

workbook.close()





# Simple Linear Regression

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('demo5.xlxs')
X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, 7].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(X_test)
print(y_pred)


