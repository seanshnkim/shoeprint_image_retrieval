import os

class Config:
    def __init__(self, loss, working_dir="", train_val_split=[0.8, 0.2]):
        self.loss_function = loss
        self.working_dir = working_dir
        self.image_labels = os.path.join(working_dir, "label_table.csv")
        self.train_val_split = train_val_split
        
        self.query_train_path = os.path.join(working_dir, "query", "train")
        self.ref_train_path = os.path.join(working_dir, "reference_v1")
        
        self.query_test_path = os.path.join(working_dir, "query", "test")
        self.ref_test_path = os.path.join(working_dir, "ref")
        
        
    def get_model_hyperparameters(self, option=0):
        #REVIEW - add more options
        if option == 0:
            return {"end_layer": 7,
                    "embedding_dim": 500,
                    "model_name": "resnet50"}
    
    
    def get_training_hyperparameters(self, option=0):
        #REVIEW - add more options
        if option == 0:
            return {
                "batch_size": 128,
                "epochs": 300,
                "early_stopping": 15,
                "margin": 5.0,
                "learning_rate": 0.0001,
                "optimizer": "Adam",
                "step_size": 5,
                "gamma": 0.5
            }
        # Result 102
        elif option == 1:
            return {
                "batch_size": 16,
                "epochs": 300,
                "early_stopping": 15,
                "margin": 2.0,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "step_size": 50,
                "gamma": 0.5}