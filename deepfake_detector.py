import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

class DeepfakeDetector:
    def __init__(self, model_path='models/model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define model architecture here (should match your trained model)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 2)
        )

        # Load the state dict
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def predict_frame(self, frame):
        # Preprocess frame
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform and add batch dimension
        frame_tensor = self.transform(frame).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(frame_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence

    def analyze_video_frames(self, frames):
        predictions = []
        confidences = []

        for frame in frames:
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred, conf = self.predict_frame(frame)
            predictions.append(pred)
            confidences.append(conf)

        # Aggregate predictions
        avg_confidence = np.mean(confidences)
        final_prediction = "FAKE" if np.mean(predictions) > 0.5 else "REAL"

        return final_prediction, avg_confidence