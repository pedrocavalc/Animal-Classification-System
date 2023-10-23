from abc import abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf

class DataLoader():
    """
    Abstract base class for data loading. It provides functionalities to load image data, 
    split it into train, validation, and test sets, and create dataloaders for deep learning.
    """
    
    def __init__(self):
        """Initializer for DataLoader class."""
        pass
    
    @abstractmethod
    def load_data(self, path: str) -> tuple:
        """
        Loads image data from the given path and splits it into train, validation, and test sets.
        
        Parameters:
        - path (str): The directory path where the image data resides.
        
        Returns:
        tuple: train_df, valid_df, test_df DataFrames.
        """
        data = {"imgpath": [], "labels": []}

        categories = os.listdir(path)
        for folder in categories:
            folder_path = os.path.join(path, folder)
            file_list = os.listdir(folder_path)
            for file in file_list:
                fpath = os.path.join(folder_path, file)
                data["imgpath"].append(fpath)
                data["labels"].append(folder)

        dataframe = pd.DataFrame(data) 
        lb = LabelEncoder()
        dataframe['encoded_labels'] = lb.fit_transform(dataframe['labels'])
        train_df, valid_df, test_df = self.__split_data(dataframe)
        return train_df, valid_df, test_df

    def __split_data(self, df_to_split: pd.DataFrame) -> tuple:
        """
        Splits the DataFrame into train, validation, and test sets.
        
        Parameters:
        - df_to_split (pd.DataFrame): The DataFrame to be split.
        
        Returns:
        tuple: train_df, valid_df, test_df DataFrames.
        """
        train_df, Temp_df = train_test_split(df_to_split,  train_size=0.70, shuffle=True, random_state=124)
        valid_df, test_df = train_test_split(Temp_df, train_size=0.70, shuffle=True, random_state=124)
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, valid_df, test_df

    @abstractmethod
    def create_dataloaders(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """
        Creates data loaders for training, validation, and testing.
        
        Parameters:
        - train_df (pd.DataFrame): DataFrame for training set.
        - valid_df (pd.DataFrame): DataFrame for validation set.
        - test_df (pd.DataFrame): DataFrame for test set.
        
        Returns:
        tuple: train_images, val_images, test_images data loaders.
        """
        generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)
        
        train_images = generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='imgpath',
            y_col='labels',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True,
            seed=42
        )

        val_images = generator.flow_from_dataframe(
            dataframe=valid_df,
            x_col='imgpath',
            y_col='labels',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=False
        )

        test_images = generator.flow_from_dataframe(
            dataframe=test_df,
            x_col='imgpath',
            y_col='labels',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=False
        )
        
        return train_images, val_images, test_images
