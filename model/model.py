import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,UpSampling2D,Multiply,Add,Cropping2D,concatenate
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import *

def crop_dim(Input,Output):
    inputx=Input.shape[-3]
    outputx=Output.shape[-2]
    
    x=int((inputx-outputx)/2)
    y=x
    
    return (x,y),(x,y)

def cnn_model():
    """
    cnn_model
    
    No inputs needed. 
    Returns model, final layer of model is DL0: (none,512,512,2)
    
    """
    #for the model we need both a path for the mask as well as the 'observed map (Q/U)' 
    
    inputs = Input(shape=(1536,1536,4))
    map_data = inputs[:,:,:,0:2]
    mask_data = inputs[:,:,:,2:]
    
    #-----nonlinear encoder path-----
    #input: (1536,1536,2)
    EN0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid', 
                   input_shape = (1536,1536,2),kernel_initializer='he_normal',activation='relu')(mask_data)
    #output: (1532,1532,16)
    EN1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN0)
    #output: (764,764,16)
    EN2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN1)
    #output: (380,380,32)
    EN3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN2)
    #output: (188,188,32)
    EN4=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',  
                   kernel_initializer='he_normal',activation='relu')(EN3)
    #output: (92,92,32)
    EN5=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN4)
    #output: (44,44,32)
    EN6=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN5)
    #output: (20,20,32)
    EN7=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN6)
    #output: (8,8,32)
    
    #-----nonlinear decoder path-----
    #Cropping2D: ((top_crop,bottom_crop),(left_crop,right_crop))
    DN7=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size=(2,2))(EN7))
    #**output: (12,12,32)**
    
    EN6C=Cropping2D(cropping=crop_dim(EN6,DN7))(EN6)
    skipcon6_NL=concatenate([EN6C,DN7],axis=3)
    DN6=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon6_NL))
    #**output: (20,20,32)**
    
    EN5C=Cropping2D(cropping=crop_dim(EN5,DN6))(EN5)
    skipcon5_NL=concatenate([EN5C,DN6],axis=3)
    DN5=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon5_NL))
    #output: (36,36,32)
    
    EN4C=Cropping2D(cropping=crop_dim(EN4,DN5))(EN4)
    skipcon4_NL=concatenate([EN4C,DN5],axis=3)
    DN4=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon4_NL))
    #output: (68,68,32)
    
    EN3C=Cropping2D(cropping=crop_dim(EN3,DN4))(EN3)
    skipcon3_NL=concatenate([EN3C,DN4],axis=3)
    DN3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon3_NL))
    #output: (132,132,32)
    
    EN2C=Cropping2D(cropping=crop_dim(EN2,DN3))(EN2)
    skipcon2_NL=concatenate([EN2C,DN3],axis=3)
    DN2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon2_NL))
    #output: (260,260,32)
    
    EN1C=Cropping2D(cropping=crop_dim(EN1,DN2))(EN1)
    skipcon1_NL=concatenate([EN1C,DN2],axis=3)
    DN1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon1_NL))
    #**output: (516,516,16)**
    
    #-----linear encoder path-----
    #stride of 2
    #input: (1536,1536,2)
    EL0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',
                   input_shape = (1536,1536,2),kernel_initializer='he_normal')(map_data)
    EL0A=Multiply()([EL0,EN0]) #EL0: (1532,1532,16); EN0: (1532,1532,16)
    #output: (1532,1532,16)
    
    EL1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL0A)
    EL1A=Multiply()([EL1,EN1]) #EL1: (764,764,16); EN1: (764,764,16)
    #output: (764,764,16)
    
    EL2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL1A)
    EL2A=Multiply()([EL2,EN2]) #EL2: (380,380,32); EN2: (380,380,32)
    #output: (380,380,32)
    
    EL3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL2A)
    EL3A=Multiply()([EL3,EN3])
    #output: (188,188,32)
    
    EL4=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL3A)
    EL4A=Multiply()([EL4,EN4])
    #output: (92,92,32)
    
    EL5=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL4A)
    EL5A=Multiply()([EL5,EN5])
    #output: (44,44,32)
    
    EL6=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL5A)
    EL6A=Multiply()([EL6,EN6])
    #output: (20,20,32)
    
    EL7=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL6A)
    #output: (8,8,32)
        
    #-----linear decoder path-----
    DL7=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(EL7))
    DL7A=Multiply()([DL7,DN7])
    #output: (12,12,32)
    
    EL6_crop=Cropping2D(cropping=crop_dim(EL6,DL7A))(EL6)
    skipcon6_L=concatenate([EL6_crop,DL7A],axis=3)
    DL6=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon6_L))
    DL6A=Multiply()([DL6,DN6])
    #output: (20,20,32)
    
    EL5_crop=Cropping2D(cropping=crop_dim(EL5,DL6A))(EL5)
    skipcon5_L=concatenate([EL5_crop,DL6A],axis=3)
    DL5=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon5_L))
    DL5A=Multiply()([DL5,DN5])
    #output: (36,36,32)
    
    EL4_crop=Cropping2D(cropping=crop_dim(EL4,DL5A))(EL4)
    skipcon4_L=concatenate([EL4_crop,DL5A],axis=3)
    DL4=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon4_L))
    DL4A=Multiply()([DL4,DN4])
    #output: (68,68,32)
    
    EL3_crop=Cropping2D(cropping=crop_dim(EL3,DL4A))(EL3)
    skipcon3_L=concatenate([EL3_crop,DL4A],axis=3)
    DL3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon3_L))
    DL3A=Multiply()([DL3,DN3])
    #output: (132,132,32)
    
    EL2_crop=Cropping2D(cropping=crop_dim(EL2,DL3A))(EL2)
    skipcon2_L=concatenate([EL2_crop,DL3A],axis=3)
    DL2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon2_L))
    DL2A=Multiply()([DL2,DN2])
    #output: (260,260,32)
    
    EL1_crop=Cropping2D(cropping=crop_dim(EL1,DL2A))(EL1)
    skipcon1_L=concatenate([EL1_crop,DL2A],axis=3)
    DL1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon1_L))
    DL1A=Multiply()([DL1,DN1])
    #output: (516,516,16)
    
    EL0_crop=Cropping2D(cropping=crop_dim(EL0,DL1A))(EL0)
    skipcon0_L=concatenate([EL0_crop,DL1A],axis=3)
    DL0=Conv2D(filters=2,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(skipcon0_L))
    #output: (512,512,2)
    #the output here will be our final map
    #droupout function can be added to prevent overtraining, generally droupout of 0.2 - 0.5 is used

    #cnn.add(Dropout(0.25))
    
    #final layer of the model itself should have a sigmoid activation function
    #final_conv= Conv2D(filters=2,1,activation='sigmoid')(DL0)
    #print(final_conv.shape)
    model = Model(inputs = inputs, outputs = DL0)
    model.compile(optimizer='Adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    
    return model

#encoder linear path at EL5 - El7 does not include multiplication layers
def cnn_model_nomult():
    """
    cnn_model
    
    No inputs needed. 
    Returns model, final layer of model is DL0: (none,512,512,2)
    
    """
    #for the model we need both a path for the mask as well as the 'observed map (Q/U)' 
    
    inputs = Input(shape=(1536,1536,4))
    map_data = inputs[:,:,:,0:2]
    mask_data = inputs[:,:,:,2:]
    
    #-----nonlinear encoder path-----
    #input: (1536,1536,2)
    EN0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid', 
                   input_shape = (1536,1536,2),kernel_initializer='he_normal',activation='relu')(mask_data)
    #output: (1532,1532,16)
    EN1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN0)
    #output: (764,764,16)
    EN2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN1)
    #output: (380,380,32)
    EN3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN2)
    #output: (188,188,32)
    EN4=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',  
                   kernel_initializer='he_normal',activation='relu')(EN3)
    #output: (92,92,32)
    EN5=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN4)
    #output: (44,44,32)
    EN6=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN5)
    #output: (20,20,32)
    EN7=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN6)
    #output: (8,8,32)
    
    #-----nonlinear decoder path-----
    #Cropping2D: ((top_crop,bottom_crop),(left_crop,right_crop))
    DN7=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size=(2,2))(EN7))
    #**output: (12,12,32)**
    
    EN6C=Cropping2D(cropping=crop_dim(EN6,DN7))(EN6)
    skipcon6_NL=concatenate([EN6C,DN7],axis=3)
    DN6=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon6_NL))
    #**output: (20,20,32)**
    
    EN5C=Cropping2D(cropping=crop_dim(EN5,DN6))(EN5)
    skipcon5_NL=concatenate([EN5C,DN6],axis=3)
    DN5=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon5_NL))
    #output: (36,36,32)
    
    EN4C=Cropping2D(cropping=crop_dim(EN4,DN5))(EN4)
    skipcon4_NL=concatenate([EN4C,DN5],axis=3)
    DN4=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon4_NL))
    #output: (68,68,32)
    
    EN3C=Cropping2D(cropping=crop_dim(EN3,DN4))(EN3)
    skipcon3_NL=concatenate([EN3C,DN4],axis=3)
    DN3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon3_NL))
    #output: (132,132,32)
    
    EN2C=Cropping2D(cropping=crop_dim(EN2,DN3))(EN2)
    skipcon2_NL=concatenate([EN2C,DN3],axis=3)
    DN2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon2_NL))
    #output: (260,260,32)
    
    EN1C=Cropping2D(cropping=crop_dim(EN1,DN2))(EN1)
    skipcon1_NL=concatenate([EN1C,DN2],axis=3)
    DN1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon1_NL))
    #**output: (516,516,16)**
    
    #-----linear encoder path-----
    #stride of 2
    #input: (1536,1536,2)
    EL0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',
                   input_shape = (1536,1536,2),kernel_initializer='he_normal')(map_data)
    EL0A=Multiply()([EL0,EN0]) #EL0: (1532,1532,16); EN0: (1532,1532,16)
    #output: (1532,1532,16)
    
    EL1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL0A)
    EL1A=Multiply()([EL1,EN1]) #EL1: (764,764,16); EN1: (764,764,16)
    #output: (764,764,16)
    
    EL2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL1A)
    EL2A=Multiply()([EL2,EN2]) #EL2: (380,380,32); EN2: (380,380,32)
    #output: (380,380,32)
    
    EL3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL2A)
    EL3A=Multiply()([EL3,EN3])
    #output: (188,188,32)
    
    EL4=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL3A)
    EL4A=Multiply()([EL4,EN4])
    #output: (92,92,32)
    
    EL5=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL4A)
    #EL5A=Multiply()([EL5,EN5])
    #output: (44,44,32)
    
    EL6=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL5)
    #EL6A=Multiply()([EL6,EN6])
    #output: (20,20,32)
    
    EL7=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL6)
    #output: (8,8,32)
        
    #-----linear decoder path-----
    DL7=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(EL7))
    DL7A=Multiply()([DL7,DN7])
    #output: (12,12,32)
    
    EL6_crop=Cropping2D(cropping=crop_dim(EL6,DL7A))(EL6)
    skipcon6_L=concatenate([EL6_crop,DL7A],axis=3)
    DL6=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon6_L))
    DL6A=Multiply()([DL6,DN6])
    #output: (20,20,32)
    
    EL5_crop=Cropping2D(cropping=crop_dim(EL5,DL6A))(EL5)
    skipcon5_L=concatenate([EL5_crop,DL6A],axis=3)
    DL5=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon5_L))
    DL5A=Multiply()([DL5,DN5])
    #output: (36,36,32)
    
    EL4_crop=Cropping2D(cropping=crop_dim(EL4,DL5A))(EL4)
    skipcon4_L=concatenate([EL4_crop,DL5A],axis=3)
    DL4=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon4_L))
    DL4A=Multiply()([DL4,DN4])
    #output: (68,68,32)
    
    EL3_crop=Cropping2D(cropping=crop_dim(EL3,DL4A))(EL3)
    skipcon3_L=concatenate([EL3_crop,DL4A],axis=3)
    DL3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon3_L))
    DL3A=Multiply()([DL3,DN3])
    #output: (132,132,32)
    
    EL2_crop=Cropping2D(cropping=crop_dim(EL2,DL3A))(EL2)
    skipcon2_L=concatenate([EL2_crop,DL3A],axis=3)
    DL2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon2_L))
    DL2A=Multiply()([DL2,DN2])
    #output: (260,260,32)
    
    EL1_crop=Cropping2D(cropping=crop_dim(EL1,DL2A))(EL1)
    skipcon1_L=concatenate([EL1_crop,DL2A],axis=3)
    DL1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon1_L))
    DL1A=Multiply()([DL1,DN1])
    #output: (516,516,16)
    
    EL0_crop=Cropping2D(cropping=crop_dim(EL0,DL1A))(EL0)
    skipcon0_L=concatenate([EL0_crop,DL1A],axis=3)
    DL0=Conv2D(filters=2,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(skipcon0_L))
    #output: (512,512,2)
    #the output here will be our final map
    #droupout function can be added to prevent overtraining, generally droupout of 0.2 - 0.5 is used

    #cnn.add(Dropout(0.25))
    
    #final layer of the model itself should have a sigmoid activation function
    #final_conv= Conv2D(filters=2,1,activation='sigmoid')(DL0)
    #print(final_conv.shape)
    model = Model(inputs = inputs, outputs = DL0)
    model.compile(optimizer='Adam',loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    
    return model

def four_model(lr=1e-4):
    """
    four_model
    
    4 layer model opposed to 8 layer original model
    No inputs needed. 
    Returns model, final layer of model is DL0: (none,512,512,2)
    
    """
    #for the model we need both a path for the mask as well as the 'observed map (Q/U)' 
    
    inputs = Input(shape=(576,576,4))
    map_data = inputs[:,:,:,0:2]
    mask_data = inputs[:,:,:,2:]
    
    #-----nonlinear encoder path-----
    #input: (576,576,2)
    EN0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid', 
                   input_shape = (576,576,2),kernel_initializer='he_normal',activation='relu')(mask_data) #output: (572,572,16)
    EN1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN0) #output: (284,284,16)
    EN2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN1) #output: (140,140,32)
    EN3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN2) #output: (68,68,32) 
    #Dropout1=Dropout(0.5)(EN3)
    
    #-----nonlinear decoder path-----
    #Cropping2D: ((top_crop,bottom_crop),(left_crop,right_crop))
    #Upsampling 2D: doubles pixels
    #Conv2D: (stride 1) removes 2x2 pixels, 2*npix - 4 per layer
    DN3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(EN3))  #replace input with EN3 with no 
                                                                                                          #dropout, output: (132,132,32)
    
    EN2C=Cropping2D(cropping=crop_dim(EN2,DN3))(EN2) #output: (132,132,32)
    skipcon2_NL=concatenate([EN2C,DN3],axis=3) 
    DN2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon2_NL))  #output: (260,260,32)
    
    EN1C=Cropping2D(cropping=crop_dim(EN1,DN2))(EN1) #output: (260,260,32)
    skipcon1_NL=concatenate([EN1C,DN2],axis=3)
    DN1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon1_NL)) #output: (516,516,16)

    #-----linear encoder path-----
    #stride of 2
    #Input: (576,576,2)
    #Conv2D layers: removes 2x2 pixels, (npix*npix - 4)/2
    EL0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',
                   input_shape = (576,576,2),kernel_initializer='he_normal')(map_data)
    EL0A=Multiply()([EL0,EN0]) #output: (572,572,16)
    
    EL1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL0A)
    EL1A=Multiply()([EL1,EN1]) #output: (284,284,16)
    
    EL2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL1A)
    EL2A=Multiply()([EL2,EN2]) #output: (140,140,32)
    
    EL3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL2A) #output:(68,68,32)
    #Dropout2=Dropout(0.5)(EL3)
    
    #-----linear decoder path-----
    DL3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(EL3)) #replace Dropout2 with EL3 if needed
    DL3A=Multiply()([DL3,DN3]) #output: (132,132,32)
    
    EL2_crop=Cropping2D(cropping=crop_dim(EL2,DL3A))(EL2) #output: (132,132,32)
    skipcon2_L=concatenate([EL2_crop,DL3A],axis=3)
    DL2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon2_L))
    DL2A=Multiply()([DL2,DN2]) #output: (260,260,32)
    
    EL1_crop=Cropping2D(cropping=crop_dim(EL1,DL2A))(EL1) #output: (260,260,32)
    skipcon1_L=concatenate([EL1_crop,DL2A],axis=3)
    DL1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon1_L))
    DL1A=Multiply()([DL1,DN1]) #output: (516,516,16)
    
    EL0_crop=Cropping2D(cropping=crop_dim(EL0,DL1A))(EL0) #output: (516,516,16)
    skipcon0_L=concatenate([EL0_crop,DL1A],axis=3)
    DL0=Conv2D(filters=2,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(skipcon0_L)) #output (512,512,2)
    
    model = Model(inputs = inputs, outputs = DL0)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    
    return model

def four_model_skip1(lr=1e-4):
    """
    four_model
    
    4 layer model opposed to 8 layer original model
    No inputs needed. 
    Returns model, final layer of model is DL0: (none,512,512,2)
    
    """
    #for the model we need both a path for the mask as well as the 'observed map (Q/U)' 
    
    inputs = Input(shape=(576,576,4))
    map_data = inputs[:,:,:,0:2]
    mask_data = inputs[:,:,:,2:]
    
    #-----nonlinear encoder path-----
    #input: (576,576,2)
    EN0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid', 
                   input_shape = (576,576,2),kernel_initializer='he_normal',activation='relu')(mask_data) #output: (572,572,16)
    EN1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN0) #output: (284,284,16)
    EN2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN1) #output: (140,140,32)
    EN3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN2) #output: (68,68,32) 
    #Dropout1=Dropout(0.5)(EN3)
    
    #-----nonlinear decoder path-----
    #Cropping2D: ((top_crop,bottom_crop),(left_crop,right_crop))
    #Upsampling 2D: doubles pixels
    #Conv2D: (stride 1) removes 2x2 pixels, 2*npix - 4 per layer
    DN3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(EN3))
                                                                                                          #dropout, output: (132,132,32)
    DN2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(DN3))  #output: (260,260,32)
    
    EN1C=Cropping2D(cropping=crop_dim(EN1,DN2))(EN1) #output: (260,260,32)
    skipcon1_NL=concatenate([EN1C,DN2],axis=3)
    DN1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon1_NL)) #output: (516,516,16)

    #-----linear encoder path-----
    #stride of 2
    #Input: (576,576,2)
    #Conv2D layers: removes 2x2 pixels, (npix*npix - 4)/2
    EL0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',
                   input_shape = (576,576,2),kernel_initializer='he_normal')(map_data)
    EL0A=Multiply()([EL0,EN0]) #output: (572,572,16)
    
    EL1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL0A)
    EL1A=Multiply()([EL1,EN1]) #output: (284,284,16)
    
    EL2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL1A)
    EL2A=Multiply()([EL2,EN2]) #output: (140,140,32)
    
    EL3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL2A) #output:(68,68,32)
    #Dropout2=Dropout(0.5)(EL3)
    
    #-----linear decoder path-----
    DL3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(EL3)) #replace Dropout2 with EL3 if needed
    DL3A=Multiply()([DL3,DN3]) #output: (132,132,32)
    
    #EL2_crop=Cropping2D(cropping=crop_dim(EL2,DL3A))(EL2) #output: (132,132,32)
    #skipcon2_L=concatenate([EL2_crop,DL3A],axis=3)
    DL2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(DL3A))
    DL2A=Multiply()([DL2,DN2]) #output: (260,260,32)
    
    EL1_crop=Cropping2D(cropping=crop_dim(EL1,DL2A))(EL1) #output: (260,260,32)
    skipcon1_L=concatenate([EL1_crop,DL2A],axis=3)
    DL1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon1_L))
    DL1A=Multiply()([DL1,DN1]) #output: (516,516,16)
    
    EL0_crop=Cropping2D(cropping=crop_dim(EL0,DL1A))(EL0) #output: (516,516,16)
    skipcon0_L=concatenate([EL0_crop,DL1A],axis=3)
    DL0=Conv2D(filters=2,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(skipcon0_L)) #output (512,512,2)
    
    model = Model(inputs = inputs, outputs = DL0)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    
    return model


def four_model_skip3(lr=1e-4):
    """
    four_model
    
    4 layer model opposed to 8 layer original model
    No inputs needed. 
    Returns model, final layer of model is DL0: (none,512,512,2)
    
    """
    #for the model we need both a path for the mask as well as the 'observed map (Q/U)' 
    
    inputs = Input(shape=(576,576,4))
    map_data = inputs[:,:,:,0:2]
    mask_data = inputs[:,:,:,2:]
    
    #-----nonlinear encoder path-----
    #input: (576,576,2)
    EN0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid', 
                   input_shape = (576,576,2),kernel_initializer='he_normal',activation='relu')(mask_data) #output: (572,572,16)
    EN1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN0) #output: (284,284,16)
    EN2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN1) #output: (140,140,32)
    EN3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN2) #output: (68,68,32) 
    #Dropout1=Dropout(0.5)(EN3)
    
    #-----nonlinear decoder path-----
    #Cropping2D: ((top_crop,bottom_crop),(left_crop,right_crop))
    #Upsampling 2D: doubles pixels
    #Conv2D: (stride 1) removes 2x2 pixels, 2*npix - 4 per layer
    skipcon3_NL=concatenate([],axis=3)
    DN3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(EN3))  #replace input with EN3 with no 
                                                                                                          #dropout, output(132,132,32)
    
    EN2C=Cropping2D(cropping=crop_dim(EN2,DN3))(EN2) #output: (132,132,32)
    skipcon2_NL=concatenate([EN2C,DN3],axis=3) 
    DN2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon2_NL))  #output: (260,260,32)
    
    EN1C=Cropping2D(cropping=crop_dim(EN1,DN2))(EN1) #output: (260,260,32)
    skipcon1_NL=concatenate([EN1C,DN2],axis=3)
    DN1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon1_NL)) #output: (516,516,16)

    #-----linear encoder path-----
    #stride of 2
    #Input: (576,576,2)
    #Conv2D layers: removes 2x2 pixels, (npix*npix - 4)/2
    EL0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',
                   input_shape = (576,576,2),kernel_initializer='he_normal')(map_data)
    EL0A=Multiply()([EL0,EN0]) #output: (572,572,16)
    
    EL1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL0A)
    EL1A=Multiply()([EL1,EN1]) #output: (284,284,16)
    
    EL2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL1A)
    EL2A=Multiply()([EL2,EN2]) #output: (140,140,32)
    
    EL3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL2A) #output:(68,68,32)
    #Dropout2=Dropout(0.5)(EL3)
    
    #-----linear decoder path-----
    DL3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(EL3)) #replace Dropout2 with EL3 if needed
    DL3A=Multiply()([DL3,DN3]) #output: (132,132,32)
    
    EL2_crop=Cropping2D(cropping=crop_dim(EL2,DL3A))(EL2) #output: (132,132,32)
    skipcon2_L=concatenate([EL2_crop,DL3A],axis=3)
    DL2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon2_L))
    DL2A=Multiply()([DL2,DN2]) #output: (260,260,32)
    
    EL1_crop=Cropping2D(cropping=crop_dim(EL1,DL2A))(EL1) #output: (260,260,32)
    skipcon1_L=concatenate([EL1_crop,DL2A],axis=3)
    DL1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon1_L))
    DL1A=Multiply()([DL1,DN1]) #output: (516,516,16)
    
    EL0_crop=Cropping2D(cropping=crop_dim(EL0,DL1A))(EL0) #output: (516,516,16)
    skipcon0_L=concatenate([EL0_crop,DL1A],axis=3)
    DL0=Conv2D(filters=2,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(skipcon0_L)) #output (512,512,2)
    
    model = Model(inputs = inputs, outputs = DL0)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    
    return model


def five_model(lr=1e-4):
    """
    four_model
    
    5 layer model opposed to 8 layer original model
    No inputs needed. 
    Returns model, final layer of model is DL0: (none,512,512,2)
    
    """
    #for the model we need both a path for the mask as well as the 'observed map (Q/U)' 
    
    inputs = Input(shape=(640,640,4))
    map_data = inputs[:,:,:,0:2]
    mask_data = inputs[:,:,:,2:]
    
    #-----nonlinear encoder path-----
    #input: (640,640,2)
    EN0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid', 
                   input_shape = (640,640,2),kernel_initializer='he_normal',activation='relu')(mask_data) #output: (636,636,16)
    EN1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN0) #output: (316,316,16)
    EN2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN1) #output: (156,156,32)
    EN3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN2) #output: (76,76,32)
    EN4=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal',activation='relu')(EN3) #output: (36,36,32)
    
    #-----nonlinear decoder path-----
    #Cropping2D: ((top_crop,bottom_crop),(left_crop,right_crop))
    DN4=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(EN4)) #output: (68,68,32)
    
    DN3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(DN4))  #output: (132,132,32)
    
    EN2C=Cropping2D(cropping=crop_dim(EN2,DN3))(EN2)
    skipcon2_NL=concatenate([EN2C,DN3],axis=3)
    DN2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon2_NL))  #output: (260,260,32)
    
    EN1C=Cropping2D(cropping=crop_dim(EN1,DN2))(EN1)
    skipcon1_NL=concatenate([EN1C,DN2],axis=3)
    DN1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal',activation='relu')(UpSampling2D(size = (2,2))(skipcon1_NL)) #output: (516,516,16)

    #-----linear encoder path-----
    #stride of 2
    #input: (640,640,2)
    EL0=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',
                   input_shape = (640,640,2),kernel_initializer='he_normal')(map_data)
    EL0A=Multiply()([EL0,EN0]) #output: (636,636,16)
    
    EL1=Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL0A)
    EL1A=Multiply()([EL1,EN1]) #output: (316,316,16)
    
    EL2=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL1A)
    EL2A=Multiply()([EL2,EN2]) #output: (156,156,32)
    
    EL3=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL2A)
    EL3A=Multiply()([EL3,EN3]) #output: (76,76,32)
    
    EL4=Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='valid',
                   kernel_initializer='he_normal')(EL3A) #output(36,36,32)
        
    #-----linear decoder path-----
    DL4=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(EL4))
    DL4A=Multiply()([DL4,DN4]) #output: (68,68,32)
    
    DL3=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(DL4A))
    DL3A=Multiply()([DL3,DN3]) #output: (132,132,32)
    
    EL2_crop=Cropping2D(cropping=crop_dim(EL2,DL3A))(EL2)
    skipcon2_L=concatenate([EL2_crop,DL3A],axis=3)
    DL2=Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon2_L))
    DL2A=Multiply()([DL2,DN2]) #output: (260,260,32)
    
    EL1_crop=Cropping2D(cropping=crop_dim(EL1,DL2A))(EL1)
    skipcon1_L=concatenate([EL1_crop,DL2A],axis=3)
    DL1=Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(skipcon1_L))
    DL1A=Multiply()([DL1,DN1]) #output: (516,516,16)
    
    EL0_crop=Cropping2D(cropping=crop_dim(EL0,DL1A))(EL0)
    skipcon0_L=concatenate([EL0_crop,DL1A],axis=3)
    DL0=Conv2D(filters=2,kernel_size=(5,5),strides=(1,1),padding = 'valid', 
               kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(skipcon0_L)) #output (512,512,2)
    
    model = Model(inputs = inputs, outputs = DL0)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,loss=tf.keras.losses.MeanSquaredError(),metrics=['accuracy'])
    
    return model
