import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

class DeepFakeClassifier(torch.nn.Module):
    def __init__(self):
        super(DeepFakeClassifier, self).__init__()
        # Simple CNN architecture for simulation
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

        self.simulation_mode = True
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.simulation_mode = False
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Falling back to simulation mode")
        else:
            print("Running in simulation mode since no model file was found")

    @torch.no_grad()
    def predict_frame(self, frame):
        """Predict a single frame."""
        try:
            if self.simulation_mode:
                # In simulation mode, use frame brightness to simulate prediction
                # Brighter frames are more likely to be classified as real
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                # Normalize brightness to [0, 1]
                normalized_brightness = avg_brightness / 255.0
                # If brightness > 0.5, predict as real (0), otherwise fake (1)
                pred = 0 if normalized_brightness > 0.5 else 1
                return pred, normalized_brightness

            # Real model prediction code
            frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            output = self.model(frame_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred = torch.argmax(probabilities, dim=1).item()
            conf = probabilities[0][pred].item()
            return pred, conf

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
            if self.simulation_mode:
                # Use majority voting for predictions
                is_fake = np.mean(predictions) > 0.5
                final_prediction = "FAKE" if is_fake else "REAL"
            else:
                # Use majority voting for real predictions
                is_fake = np.mean(predictions) > 0.5
                final_prediction = "FAKE" if is_fake else "REAL"

            return final_prediction, avg_confidence

        except Exception as e:
            print(f"Error in analyze_video_frames: {str(e)}")
            return "ERROR", 0.0