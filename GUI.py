# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil #module helps you automate copying files and directories
import csv
import numpy as np
from PIL import Image, ImageTk #module to open, manipulate and save images
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window = tk.Tk()
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
#window.geometry('1280x720')
window.configure(background='white')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(window, text="Attendance Using Face Recogniton" ,fg="blue"  ,width=42  ,height=3,font=('times', 30, 'bold')) 

message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="red",bg="grey90" ,font=('times', 15, ' bold ') ) 
lbl.place(x=300, y=200)

txt = tk.Entry(window,width=20  ,fg="red",bg="grey90",font=('times', 15, ' bold '))
txt.place(x=600, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="red",bg="grey90",height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=300, y=300)

txt2 = tk.Entry(window,width=20 ,fg="red",bg="grey90",font=('times', 15, ' bold ')  )
txt2.place(x=600, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red",bg="grey85" ,height=2 ,font=('times', 15, ' bold')) 
lbl3.place(x=300, y=400)

message = tk.Label(window, text=""  ,fg="red" ,bg="grey90" ,width=50  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=600, y=400)

lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="red",bg="grey90" ,height=2 ,font=('times', 15, ' bold')) 
lbl3.place(x=300, y=650)


message2 = tk.Label(window, text="" ,fg="red" ,bg="grey90"  ,activeforeground = "green",width=50  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=600, y=650)
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    path1 = "dataset/train/"
    path2="dataset/test/"
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        try:
            # Create target Directory
            os.mkdir(path1+str(name)+" "+str(Id))
            os.mkdir(path2+str(name)+" "+str(Id))
            print("Directory " , path1+str(name)+" "+str(Id),  " Created ")
            print("Directory " , path2+str(name)+" "+str(Id),  " Created ")
        except FileExistsError:
            print("Directory " , path1+str(name) ,  " already exists")
            print("Directory " , path2+str(name),  " already exists")   
        sampleNum=0
        while(True):
            ret, img = cam.read()
            frame = img.copy()
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                if(sampleNum<=7):
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite(path1+str(name)+" "+str(Id)+ "\\"+name+"."+Id+"."+str(sampleNum)+ ".jpg", frame[y:y+h, x:x+w])
                elif(sampleNum>7 and sampleNum<=10):
                    cv2.imwrite(path2+str(name)+" "+str(Id)+ "\\"+name+"."+Id+"."+str(sampleNum)+ ".jpg", frame[y:y+h, x:x+w])
                #display the frame
                cv2.imshow('img',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>10:
                break
        cam.release()
        cv2.destroyAllWindows()
    
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    import image_crop
    import embed
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def TrackImages():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    path = ""# path were u want store the data set
    id = "image"
    try:
        # Create target Directory
        os.mkdir(path+str(id))
        print("Directory " , path+str(id),  " Created ")
    except FileExistsError:
        print("Directory " , path+str(id) ,  " already exists")
    sampleN=0

    while 1:
        ret, img = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            sampleN=sampleN+1
            cv2.imwrite(path+str(id)+ "\\" +str(sampleN)+ ".jpg", img[y:y+h, x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.waitKey(100)
        cv2.imshow('img',img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        # break if the sample number is morethan 100
        elif sampleN>0:
            break
    cap.release()
    cv2.destroyAllWindows()
    from random import choice
    from numpy import load, asarray
    from numpy import expand_dims
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import SVC
    from matplotlib import pyplot
    from facedemo import extract_face, get_embedding
    from keras.models import load_model
    # load faces
    data = load('faces-dataset.npz')
    testX_faces = data['arr_2']
    # load face embeddings
    data = load('faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    models = load_model('facenet_keras.h5')
    print("Test start")
    # test model on a random example from the test dataset
    arface = extract_face('image/1.jpg')
    em = get_embedding(models, arface)
    list(em)
    print("test on embedded")
    newT = asarray(em)
    print(newT.shape)
    
    # prediction for the face
    samples = expand_dims(newT, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    # plot
    if(class_probability>99.98):
        pyplot.imshow(arface)
        title = '%s (%.3f)' % (predict_names[0], class_probability)
        pyplot.title(title)
        #pyplot.show()
        df=pd.read_csv("StudentDetails\StudentDetails.csv")        
        col_names =  ['Roll_No.','Name','Date','Time']
        attendance = pd.DataFrame(columns = col_names)
        ts = time.time()      
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        aa=predict_names[0]
        name=[word for word in aa.split() if word.isalpha()]
        num= [int(word) for word in aa.split() if word.isdigit()]
        attendance.loc[len(attendance)] = [num[0],name[0],date,timeStamp]
        print([num[0],name[0],date,timeStamp])
        Hour,Minute,Second=timeStamp.split(":")
        fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        attendance.to_csv(fileName,index=False)
        res=attendance
        message2.configure(text= res)
        pyplot.show()
        
    else:
        pyplot.imshow(arface)
        title = '%s' % ("Unknown")
        pyplot.title(title)
        pyplot.show()
        noOfFile=len(os.listdir("ImagesUnknown"))+1
        cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", img[y:y+h,x:x+w])
        res="Unknown Person"
        message2.configure(text= res)
    
        
   
    #attendance=attendance.drop_duplicates(subset=['Name'],keep='first')     
    print(res)
    #res="Attendance marked"
    #message2.configure(text= res)

  
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="red",width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=850, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="red",width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=850, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="red",width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red",width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="red",width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red",width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

 
window.mainloop()
