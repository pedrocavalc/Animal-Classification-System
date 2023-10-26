from components.model.resnet import CustomResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
mlflow.set_tracking_uri('http://mlflow-server:5000')

class TrainOrchestrator:
    def __init__(self) -> None:
        self.client = mlflow.MlflowClient()

    def run(self, train_data, test_data, valid_data):
        num_classes = len(set(train_data.classes))
        model = CustomResNet50(num_classes=num_classes).get_model()
        model.summary()
        mlflow.set_experiment("ResNet50")
        mlflow.tensorflow.autolog()
        with mlflow.start_run():
            history = model.fit(
            train_data,
            steps_per_epoch=len(train_data),
            validation_data=valid_data,
            validation_steps=len(valid_data),
            epochs=20,
            callbacks=[
                EarlyStopping(monitor = "val_loss", 
                                    patience = 3,
                                    restore_best_weights = True), 
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='min') 
            ]
        )
            self.test_register_model(model, test_data)
    
    def test_register_model(self, model,test_data):
        model_lists = self.client.search_registered_models()
        if not model_lists:
            run_id = mlflow.active_run().info.run_id
            artifact_uri = self.client.get_run(run_id).info.artifact_uri
            model_path = f"{artifact_uri}/{run_id}/model"
            self.client.create_registered_model(name='ResNet50')
            self.client.create_model_version(run_id=run_id, name='ResNet50', source=model_path)
            self.client.transition_model_version_stage(
                 name='ResNet50',
                 version=1,
                 stage='Production',
                 archive_existing_versions=True
             )
  
        else:
            acc = model.evaluate(test_data, steps=len(test_data))[1]
            if acc > 0.4:
                """
                TODO: implemente a logic to register the model
                """
                models = self.client.get_registered_model(name='ResNet50')
                existing_versions = len(models.latest_versions)
                run_id = mlflow.active_run().info.run_id
                artifact_uri = self.client.get_run(run_id).info.artifact_uri
                model_path = f"{artifact_uri}/{run_id}/model"
                self.client.create_model_version(run_id=run_id, name='ResNet50', source=model_path)
                self.client.transition_model_version_stage(
                 name='ResNet50',
                 version=existing_versions + 1,
                 stage='Production',
                 archive_existing_versions=True
             )
                

       
