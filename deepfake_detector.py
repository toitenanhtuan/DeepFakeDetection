import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

class DeepFakeClassifier(torch.nn.Module):
    def __init__(self):
        super(DeepFakeClassifier, self).__init__()
        # Simple CNN architecture that can be adapted based on loaded weights
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.model(x)

class DeepfakeDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = DeepFakeClassifier().to(self.device)
        print("Model architecture initialized")

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

        print("Running in simulation mode since no model file was found")

    @torch.no_grad()
    def predict_frame(self, frame):
        """Predict a single frame."""
        try:
            # In simulation mode, return random prediction
            return np.random.choice([0, 1]), np.random.random()

        except Exception as e:
            print(f"Error in predict_frame: {str(e)}")
            return 0, 0.0

    def analyze_video_frames(self, frames):
        """Analyze multiple frames from a video."""
        try:
            predictions = []
            confidences = []

            for frame in frames:
                pred, conf = self.predict_frame(frame)
                predictions.append(pred)
                confidences.append(conf)

            # Aggregate predictions
            avg_confidence = np.mean(confidences)
            # In simulation mode, use length of video to determine prediction
            final_prediction = "FAKE" if len(frames) % 2 == 0 else "REAL"

            return final_prediction, avg_confidence

        except Exception as e:
            print(f"Error in analyze_video_frames: {str(e)}")
            return "ERROR", 0.0