import argparse
import torch

class Config:
    def __init__(self):
        # Argument Parser for overriding config values
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_csv', type=str, default='training_obesity_df.csv', help='Path to the training CSV file')
        parser.add_argument('--val_csv', type=str, default='validation_obesity_df.csv', help='Path to the validation CSV file')
        parser.add_argument('--test_csv', type=str, default='test_obesity_df.csv', help='Path to the test CSV file')
        parser.add_argument('--model_type', type=str, default='efficientnet_b1', help='efficientnet_b1, convnext, vit')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs for training')
        parser.add_argument('--save_model_path', type=str, default='best_model.pth', help='Path to save the best model')
        parser.add_argument('--early_stop_patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped')
        args = parser.parse_args()

        # Data paths
        self.train_csv = args.train_csv  # Path to the training CSV file
        self.val_csv = args.val_csv  # Path to the validation CSV file
        self.test_csv = args.test_csv  # Path to the test CSV file

        # Model configuration
        self.cnn_model_name = args.model_type  # CNN model to use (resnet50, efficientnet_b0, densenet121, inception_v3, mobilenet_v2)
        self.num_classes = 3  # Number of output classes (e.g., underweight, normal, obese)

        # Training parameters
        self.batch_size = args.batch_size  # Batch size for training
        self.learning_rate = args.learning_rate  # Learning rate for optimizer
        self.num_epochs = args.num_epochs  # Number of epochs for training
        self.save_model_path = args.save_model_path  # Path to save the best model
        self.early_stop_patience = args.early_stop_patience  # Early stopping patience

        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
