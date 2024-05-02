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
from PIL import ImageTk

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
        mcp = tf.keras.callbacks.ModelCheckpoint('CNN.h5', save_best_only=True, mode='auto', monitor='val_accuracy')
        es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=3)
        CNN_history = self.CNN.fit(self.train_dataset,validation_data=self.valid_dataset, epochs = 36,verbose = 1, callbacks=[lrd,mcp,es], shuffle=True)
        CNN_scores = self.CNN.evaluate(self.selftest_dataset, verbose=1)
        self.plot_history(self.CNN_history, 'CNN')
        self.save_history_plot(CNN_history, 'CNN', 'CNN_history_plot.png')
    
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
        
    # def resize_image(self, image_path, width, height):
    #     # Open the image file
    #     image = PIL.Image.open(image_path)
    #     # Resize the image to fit the specified width and height
    #     resized_image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
    #     # Convert the resized image to a Tkinter PhotoImage
    #     return ImageTk.PhotoImage(resized_image)
    
    def show_text(self, text, frame):
        pass
    
    # def show_image(self):
    #     self.image_label.pack_forget()
    #     resized_image = PIL.Image.open(self.image_path)
    #     # .resize((self.width, self.height), PIL.Image.Resampling.LANCZOS)
    #     self.image_label = Label(self.right_frame, image=ImageTk.PhotoImage(resized_image))
    #     self.image_label.pack(fill="both", expand=True)
        
    #     # self.image_label = Label(self.right_frame, image=image)
    #     # self.image_label.image = image  # Keep a reference to the image to prevent garbage collection
        
        
    def __init__(self,root):
        self.root=root
        #window size
        self.root.geometry("1050x600+0+0")
        self.root.title("Lung Cancer Detection")  # title of the GUI window
        self.root.config(bg="royal blue")  # specify background color
        # img4=img4.resize((1006,500),Image.ANTIALIAS)
        # self.photoimg4=ImageTk.PhotoImage(img4)

        #Title
        title_lbl=Label(root, text="Lung Cancer Detection",font=("Courier New",30,"bold"),bg="royal blue",fg="black",)
        title_lbl.grid(row=0, column=0, columnspan=2, padx=10, pady=5)
                
        #LEFT Frame
        self.left_frame = Frame(root, height=600, bg='royal blue')
        self.left_frame.grid(row=1, column=0, padx=10, pady=5)
        
        self.model = Lung_Cancer_Model()
        # Buttons
        button_pady = 10
        b1=Button(self.left_frame,relief=RAISED, width=20, text="PreprocesingData",cursor="hand2",command=self.model.preprocessing,font=("Arial",15,"bold"),bg="white",fg="black")
        b1.grid(row=1, column=0, padx=50, pady=button_pady)
        b2=Button(self.left_frame,relief=RAISED, width=20, text="Traning Data",cursor="hand2",command=self.model.training,font=("Arial",15,"bold"),bg="white",fg="black")
        b2.grid(row=2, column=0, padx=50, pady=button_pady)
        b3=Button(self.left_frame,relief=RAISED, width=20, text="Prediction",cursor="hand2",command=self.model.prediction,font=("Arial",15,"bold"),bg="white",fg="black")
        b3.grid(row=3, column=0, padx=50, pady=button_pady)
        b4=Button(self.left_frame,relief=RAISED, width=20, text="Show Result",cursor="hand2",command=print(),font=("Arial",15,"bold"),bg="white",fg="black")
        b4.grid(row=4, column=0, padx=50, pady=button_pady)
        b1.bind("<Enter>", self.on_enter) 
        b1.bind("<Leave>", self.on_leave)
        b2.bind("<Enter>", self.on_enter) 
        b2.bind("<Leave>", self.on_leave)
        b3.bind("<Enter>", self.on_enter) 
        b3.bind("<Leave>", self.on_leave)
        b4.bind("<Enter>", self.on_enter) 
        b4.bind("<Leave>", self.on_leave)
        
        # Create frames and labels in left_frame
        # Label(left_frame, text='', bg='royal blue').grid(row=5, column=0, padx=5, pady=100)
        Label(self.left_frame, text="text").grid(row=6, column=0, padx=5, pady=100)
        
        #RIGHT Frame
        frame_width = 400
        frame_height = 420

        # Create the frame
        self.right_frame = Frame(root, width=frame_width, height=frame_height, bg='grey')
        self.right_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # Path to the image file
        image_path = "D:\code\AI\Lung cancer\lung\LungImages.png"
        my_image = ImageTk.PhotoImage(PIL.Image.open(image_path))
        image_label = Label(self.right_frame, image=my_image)
        image_label.grid(row=1, column=1,columnspan=2)
        
        # Configure row and column weights to make the sub-frame expandable
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
if __name__ == "__main__":
    root=Tk()
    obj=LCD_CNN(root)
    root.mainloop()
