from components.model.resnet import CustomResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
mlflow.set_tracking_uri('http://0.0.0.0:5000')

class TrainOrchestrator:
    def __init__(self) -> None:
        self.client = mlflow.MlflowClient()

    def run(self, train_data, test_data, valid_data):
        num_classes = len(set(train_data.classes))
        model = CustomResNet50(num_classes=num_classes).get_model()
        model.summary()
        mlflow.set_experiment("ResNet50")
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
            self.test_register_model(model, test_data)
    
    def test_register_model(self, model,test_data):
        acc = model.evaluate(test_data, steps=len(test_data))[1]
        model_in_production = self.client.get_latest_versions("test", stages=["Production"])
        print(model_in_production)
       
