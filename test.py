import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from IPython.display import Image
import tensorflow.keras.backend as K
import splitfolders
import pandas as pd
import numpy as np
import seaborn as sns

from tkinter import *
import PIL.Image
from PIL import Image, ImageTk

class Lung_Cancer_Model:
    def __init__(self):
        self.DIR = 'rawData'
        self.CATS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
        self.DEST_DIR = "data"
        self.FINAL_DIR = 'processedData'
        self.METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.Recall(name='Recall'),  
        ]
        self.BATCH_SIZE=32
        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split = 0.2,
                                                                        rotation_range=5,
                                                                        width_shift_range=0.2,
                                                                        height_shift_range=0.2,
                                                                        shear_range=0.2,
                                                                        horizontal_flip=True,
                                                                        vertical_flip=True,
                                                                        fill_mode='nearest'
                                                                        )
        self.valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
        self.test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
        self.train_dataset  = self.train_datagen.flow_from_directory(directory = 'processedData/train',
                                                    target_size = (224,224),
                                                    class_mode = 'binary',
                                                    batch_size = 32)
        self.valid_dataset = self.valid_datagen.flow_from_directory(directory = 'processedData/val',
                                                    target_size = (224,224),
                                                    class_mode = 'binary',
                                                    batch_size = 32)
        self.test_dataset = self.test_datagen.flow_from_directory(directory = 'processedData/test',
                                                    target_size = (224,224),
                                                    class_mode = 'binary',
                                                    batch_size = 32)
        
        self.CNN = tf.keras.Sequential()
        self.CNN_history = None
        self.batch = next(self.test_dataset)
    
    def preprocessing(self):
        if not os.path.exists(self.DEST_DIR):
            os.makedirs('data')
            os.makedirs("data/cancerous")
            os.makedirs("data/non-cancerous")
        for category in self.CATS:
            path = os.path.join(self.DIR, category)
            for image in os.listdir(path):
                curr = os.path.join(path, image)
                img = cv2.imread(curr, 0)
                equalizedImage = cv2.equalizeHist(img)
                e, segmentedImage = cv2.threshold(equalizedImage, 128, 255, cv2.THRESH_TOZERO)
                if category == 'normal':
                    imgDest = curr.replace('rawData/normal', 'data/non-cancerous')
                    cv2.imwrite(imgDest, segmentedImage)
                else:
                    imgDest = curr.replace('rawData/adenocarcinoma', 'data/cancerous')
                    imgDest = imgDest.replace('rawData/large.cell.carcinoma', 'data/cancerous')
                    imgDest = imgDest.replace('rawData/squamous.cell.carcinoma', 'data/cancerous')
                    cv2.imwrite(imgDest, segmentedImage)
        print("Processed data directory created successfully at", self.DEST_DIR)
        print("SUCCESS!")
    
    #Neural net
    def cnn(self):
        self.CNN.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
        self.CNN.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation='relu'))
        self.CNN.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.CNN.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.CNN.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.CNN.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        self.CNN.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.CNN.add(tf.keras.layers.Dropout(rate=0.25))
        self.CNN.add(tf.keras.layers.Flatten())
        self.CNN.add(tf.keras.layers.Dense(units=64, activation='relu'))
        self.CNN.add(tf.keras.layers.Dropout(rate=0.25))
        self.CNN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        self.CNN.compile(optimizer='adam',
                    loss=tf.keras.losses.binary_crossentropy, metrics=self.METRICS)  
         
    def training(self):
        self.cnn()
        lrd = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',patience = 3,verbose = 1,factor = 0.50, min_lr = 1e-7)
        mcp = tf.keras.callbacks.ModelCheckpoint('CNN.keras', save_best_only=True, mode='auto', monitor='val_accuracy')
        es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=3)
        self.CNN_history = self.CNN.fit(self.train_dataset,validation_data=self.valid_dataset, epochs = 36,verbose = 1, callbacks=[lrd,mcp,es], shuffle=True)
        CNN_scores = self.CNN.evaluate(self.test_dataset, verbose=1)
        self.plot_history(self.CNN_history, 'CNN')
        self.save_history_plot(self.CNN_history, 'CNN', 'results/CNN_history_plot.png')

    def plot_history(self, hist, name):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(hist.history['Accuracy'])
        axs[0].plot(hist.history['val_Accuracy'])
        axs[0].set_title(f'{name} Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['train', 'val', 'F1', 'Recall'], loc='upper left')

        axs[1].plot(hist.history['loss'])
        axs[1].plot(hist.history['val_loss'])
        axs[1].set_title(f'{name} Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['train', 'val'], loc='upper left')

        axs[2].plot(hist.history['Precision'])
        axs[2].plot(hist.history['val_Precision'])
        axs[2].set_title(f'{name} Precision')
        axs[2].set_ylabel('Precision')
        axs[2].set_xlabel('Epoch')
        axs[2].legend(['train', 'val'], loc='upper left')

        plt.show()

    def save_history_plot(self, hist, name, filename):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(hist.history['accuracy'])
        axs[0].plot(hist.history['val_accuracy'])
        axs[0].set_title(f'{name} Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['train', 'val'], loc='upper left')

        axs[1].plot(hist.history['loss'])
        axs[1].plot(hist.history['val_loss'])
        axs[1].set_title(f'{name} Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['train', 'val'], loc='upper left')

        axs[2].plot(hist.history['precision'])
        axs[2].plot(hist.history['val_precision'])
        axs[2].set_title(f'{name} Precision')
        axs[2].set_ylabel('Precision')
        axs[2].set_xlabel('Epoch')
        axs[2].legend(['train', 'val'], loc='upper left')

        # Save the plot as an image file
        plt.savefig(filename)  

    def predAll(self, batch, i):
        label_dict = {0: 'Cancerous', 1: 'Non-Cancerous'}
        cnnPred = 0 if self.CNN.predict(batch[0][i].reshape(-1, 224, 224, 3)) < 0.5 else 1
        return f'CNN: {label_dict[cnnPred]}, Truth: {label_dict[int(batch[1][i])]}'

    def prediction(self):
        plt.figure(figsize=(20, 20))
        ax1 = plt.subplot(2, 2, 1)
        plt.imshow(self.batch[0][0])
        label = self.predAll(self.batch, 0)
        ax1.set_title(label)
        
        ax2 = plt.subplot(2, 2, 2)
        plt.imshow(self.batch[0][5])
        label = self.predAll(self.batch, 1)
        ax2.set_title(label)

        ax3 = plt.subplot(2, 2, 3)
        plt.imshow(self.batch[0][2])
        label = self.predAll(self.batch, 2)
        ax3.set_title(label)

        ax4 = plt.subplot(2, 2, 4)
        plt.imshow(self.batch[0][3])
        label = self.predAll(self.batch, 3)
        ax4.set_title(label)
        plt.suptitle('Predicted vs True labels')
        plt.show()
    
        
class LCD_CNN:     
    def on_enter(self, event):
        event.widget.config(bg="maroon1", fg="white")

    def on_leave(self, event):
        event.widget.config(bg="white", fg="black")

    def show_text(self, text, frame):
        pass        
        
    def __init__(self,root):
        self.root=root
        
        #Config GUI Window
        self.root.geometry("1050x600+0+0")
        self.root.title("Lung Cancer Detection")  # title of the GUI window
        self.root.config(bg="royal blue")  # specify background color

        #Title
        title_lbl=Label(root, text="Lung Cancer Detection",font=("Courier New",30,"bold"),bg="royal blue",fg="black",)
        title_lbl.grid(row=0, column=0, columnspan=2, padx=10, pady=5)
        
        #Start model
        self.model = Lung_Cancer_Model()
        
        # Buttons
        button_pady = 10
        self.b1=Button(relief=RAISED, width=20, text="PreprocesingData",cursor="hand2",command=self.funtion1,font=("Arial",15,"bold"),bg="white",fg="black")
        self.b1.place(x=30, y=90)
        self.b2=Button(relief=RAISED, width=20, text="Traning Data",cursor="hand2",command=self.model.training,font=("Arial",15,"bold"),bg="white",fg="black")
        self.b2.place(x=30, y=150)
        self.b3=Button(relief=RAISED, width=20, text="Prediction",cursor="hand2",command=self.model.prediction,font=("Arial",15,"bold"),bg="white",fg="black")
        self.b3.place(x=30, y=210)
        self.b4=Button(relief=RAISED, width=20, text="Show Result",cursor="hand2",command=print(),font=("Arial",15,"bold"),bg="white",fg="black")
        self.b4.place(x=30, y=270)
        self.b1.bind("<Enter>", self.on_enter) 
        self.b1.bind("<Leave>", self.on_leave)
        self.b2.bind("<Enter>", self.on_enter) 
        self.b2.bind("<Leave>", self.on_leave)
        self.b3.bind("<Enter>", self.on_enter) 
        self.b3.bind("<Leave>", self.on_leave)
        self.b4.bind("<Enter>", self.on_enter) 
        self.b4.bind("<Leave>", self.on_leave)
        
        # Lable to show state 
        # Label(text="text").grid(row=6, column=0, padx=5, pady=100)
        
        # Show image
        image1 = Image.open("LungImages.png")
        image1=image1.resize((700,500), PIL.Image.Resampling.LANCZOS)
        test = ImageTk.PhotoImage(image1)
        self.label1 = Label(image=test)
        self.label1.image = test
        # Position image
        self.label1.place(x=300, y=80)
        # label1.place_forget()
        self.label1.image 
        
    def funtion1(self):
        # Run model procprocessing
        self.model.preprocessing()
        
        #Run test processing method in a image
        img = cv2.imread('rawData/squamous.cell.carcinoma/squamous.cell.carcinoma1.png', 0)
        equalizedImage = cv2.equalizeHist(img)
        e, segmentedImage = cv2.threshold(equalizedImage, 128, 255, cv2.THRESH_TOZERO)
        plt.figure(figsize=(20, 6))
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Raw image')
        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(equalizedImage, cv2.COLOR_BGR2RGB))
        ax2.set_title('Equalized image')
        ax3 = plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(segmentedImage, cv2.COLOR_BGR2RGB))
        ax3.set_title('Equalized & Segmented image')
        plt.suptitle('Preprocessing Example')
        
        plt.savefig("results/preprocessing.png")
        
        self.showimage("results/preprocessing.png")
        
        self.b1.bind("<Enter>", self.on_leave)
        self.b1["state"] = "disabled"
        # self.b2.config(cursor="arrow") 
        # self.b3["state"] = "normal"
        # self.b3.config(cursor="hand2")
        
    def showimage(self, image_path):
        self.label1.place_forget()
        image1 = Image.open(image_path)
        image1=image1.resize((700,400), PIL.Image.Resampling.LANCZOS)
        test = ImageTk.PhotoImage(image1)
        self.label1 = Label(image=test)
        self.label1.image = test
        # Position image
        self.label1.place(x=300, y=70)
        
if __name__ == "__main__":
    root=Tk()
    obj=LCD_CNN(root)
    root.mainloop()
 