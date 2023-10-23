from components.model.resnet import CustomResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow

class TrainOrchestrator:
    def __init__(self) -> None:
        pass

    def run(self, train_data, test_data, valid_data):
        num_classes = len(set(train_data.classes))
        model = CustomResNet50(num_classes=num_classes).get_model()
        model.summary()
        mlflow.autolog()
        with mlflow.start_run():
            history = model.fit(
            train_data,
            steps_per_epoch=len(train_data),
            validation_data=valid_data,
            validation_steps=len(valid_data),
            epochs=1,
            callbacks=[
                EarlyStopping(monitor = "val_loss", 
                                    patience = 3,
                                    restore_best_weights = True), 
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='min') 
            ]
        )
    
