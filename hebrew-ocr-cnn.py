import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import random
import glob


class HebrewOCR:
    def __init__(self, data_dir, img_height=32, img_width=32):
        """
        Initialize the Hebrew OCR model
        
        Args:
            data_dir: Directory containing the labeled dataset
            img_height: Target image height for the model
            img_width: Target image width for the model
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.class_names = []
        self.history = None
        
        # Hebrew alphabet mapping (for reference)
        self.hebrew_letters = {
            0: 'ב', 1: 'כ', 2: 'פ', 3: 'ך', 4: 'ה', 5: 'ר', 6: 'ף', 7: 'ח', 8: 'מיקס',
            9: 'צ', 10: 'ק', 11: 'ט', 12: 'נ', 13: 'ת', 14: 'ג', 15: 'ן', 16: 'ם',
            17: 'ע', 18: 'י', 19: 'ד', 20: 'חצי-ק', 21: 'זבל', 22: 'ל', 23: 'א', 24: 'ז',
            25: 'ש', 26: 'זבל', 27: 'לאידוע', 28: 'ו', 29: 'מ', 30: 'ץ', 31: 'ס'
        }
        #self.hebrew_letters = {
        #    0: 'א', 1: 'ב', 2: 'ג', 3: 'ד', 4: 'ה', 5: 'ו', 6: 'ז', 7: 'ח', 8: 'ט',
        #    9: 'י', 10: 'כ', 11: 'ך', 12: 'ל', 13: 'מ', 14: 'ם', 15: 'נ', 16: 'ן',
        #    17: 'ס', 18: 'ע', 19: 'פ', 20: 'ף', 21: 'צ', 22: 'ץ', 23: 'ק', 24: 'ר',
        #    25: 'ש', 26: 'ת'
        #}
    
    def load_dataset(self, validation_split=0.2, test_split=0.1, batch_size=32):
        """
        Load the Hebrew letters dataset and prepare it for training
        
        Args:
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            batch_size: Batch size for training
            
        Returns:
            train_ds, val_ds, test_ds: TensorFlow dataset objects
        """
        print("Loading and preparing dataset...")
        
        # Load images and labels
        images = []
        labels = []
        self.class_names = []
        
        # Walk through the data directory to load images and create labels
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path) and not 'vavyoud' in class_path:
                self.class_names.append(class_dir)
                class_idx = len(self.class_names) - 1
                
                for img_path in glob.glob(os.path.join(class_path, "*.jpg")) + \
                                glob.glob(os.path.join(class_path, "*.png")):
                    try:
                        # Load image and convert to grayscale
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (self.img_width, self.img_height))
                        
                        # Normalize pixel values
                        img = img.astype(np.float32) / 255.0
                        
                        # Add channel dimension (TensorFlow expects this)
                        img = np.expand_dims(img, axis=-1)
                        
                        images.append(img)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=validation_split + test_split, stratify=y, random_state=42
        )
        
        # Further split temp data into validation and test sets
        split_ratio = test_split / (validation_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=split_ratio, stratify=y_temp, random_state=42
        )
        
        # Report dataset sizes
        print(f"Dataset loaded: {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        
        # Store test data for later evaluation
        self.test_images = X_test
        self.test_labels = y_test
        
        return train_ds, val_ds, test_ds
    
    def build_model(self):
        """
        Build a CNN model for Hebrew letter recognition
        """
        print("Building model...")
        
        num_classes = len(self.class_names)
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_height, self.img_width, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def train(self, train_ds, val_ds, epochs=50, patience=10):
        """
        Train the model on the provided dataset
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Maximum number of epochs to train
            patience: Number of epochs with no improvement before early stopping
            
        Returns:
            History object containing training metrics
        """
        if self.model is None:
            self.build_model()
        
        print("Training model...")
        
        # Set up callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks_list
        )
        
        self.history = history
        return history
    
    def evaluate(self, test_ds):
        """
        Evaluate the model on the test dataset
        
        Args:
            test_ds: Test dataset
            
        Returns:
            test_loss, test_accuracy
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        print("Evaluating model...")
        test_loss, test_accuracy = self.model.evaluate(test_ds)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        predictions = []
        true_labels = []
        
        for x, y in test_ds:
            preds = self.model.predict(x)
            pred_classes = np.argmax(preds, axis=1)
            predictions.extend(pred_classes)
            true_labels.extend(y.numpy())
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                   target_names=[f"{i}: {self.hebrew_letters.get(i, str(i))}" for i in range(len(self.class_names))]))
        
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.hebrew_letters.get(i, str(i)) for i in range(len(self.class_names))],
                   yticklabels=[self.hebrew_letters.get(i, str(i)) for i in range(len(self.class_names))])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """
        Plot the training and validation accuracy/loss
        """
        if self.history is None:
            print("Model has not been trained yet")
            # raise ValueError("Model has not been trained yet")
        
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved as 'training_history.png'")

    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save class names
        with open(f"{filepath}_classes.txt", 'w', encoding='utf-8') as f:
            for i, class_name in enumerate(self.class_names):
                f.write(f"{i},{class_name},{self.hebrew_letters.get(i, '')}\n")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
        """
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Load class names
        self.class_names = []
        if os.path.exists(f"{filepath}_classes.txt"):
            with open(f"{filepath}_classes.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    self.class_names.append(parts[1])
    
    def visualize_predictions(self, num_samples=10):
        """
        Visualize some predictions on the test set
        
        Args:
            num_samples: Number of samples to visualize
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self, 'test_images') or len(self.test_images) == 0:
            raise ValueError("No test data available")
        
        # Select random samples
        indices = random.sample(range(len(self.test_images)), min(num_samples, len(self.test_images)))
        
        # Make predictions
        samples = self.test_images[indices]
        true_labels = self.test_labels[indices]
        predictions = self.model.predict(samples)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Plot
        plt.figure(figsize=(15, num_samples * 2))
        for i, idx in enumerate(indices):
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(self.test_images[idx].squeeze(), cmap='gray')
            plt.title(f"True: {self.hebrew_letters.get(true_labels[i], str(true_labels[i]))}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(self.test_images[idx].squeeze(), cmap='gray')
            plt.title(f"Pred: {self.hebrew_letters.get(pred_labels[i], str(pred_labels[i]))}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.bar(range(len(self.class_names)), predictions[i])
            plt.xticks([])
            plt.title('Confidence')
        
        plt.tight_layout()
        plt.savefig('prediction_samples.png')
        print("Prediction samples saved as 'prediction_samples.png'")
    
    def process_document(self, image_path, min_size=20, padding=5):
        """
        Process a document image to recognize Hebrew letters
        
        Args:
            image_path: Path to the document image
            min_size: Minimum size of contours to consider
            padding: Padding to add around detected characters
            
        Returns:
            processed_image: Image with recognized letters highlighted
            text: Recognized text
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create a copy for drawing results
        result_image = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours right-to-left (for Hebrew reading order)
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0], reverse=True)
        
        recognized_chars = []
        
        # Process each contour
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise
            if w < min_size or h < min_size:
                continue
            
            # Extract character ROI with padding
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)
            
            roi = gray[y_start:y_end, x_start:x_end]

            def switch_colors(image):
                # Invert the colors of the image
                return cv2.bitwise_not(image)

            roi = switch_colors(roi)
            # Resize to model input size
            roi_resized = cv2.resize(roi, (self.img_width, self.img_height))
            
            # Normalize
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            
            # Add channel dimension and batch dimension
            roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)
            
            # Make prediction
            pred = self.model.predict(roi_input)[0]
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]
            
            # Get Hebrew letter
            letter = self.hebrew_letters.get(pred_class, str(pred_class))
            recognized_chars.append(letter)
            
            # Draw bounding box and label
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, f"{letter} ({confidence:.2f})", (x, y-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Combine characters into text (right-to-left)
        text = ''.join(recognized_chars)
        
        return result_image, text


def prepare_example_dataset(output_dir):
    """
    Creates a minimal example dataset for demonstration purposes
    
    Args:
        output_dir: Directory to save the example dataset
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Hebrew letters to include in the demo
    letters = ['א', 'ב', 'ג', 'ד', 'ה', 'ו']
    
    # Create directories for each letter
    for i, letter in enumerate(letters):
        letter_dir = os.path.join(output_dir, str(i))
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
        
        # Create synthetic images for this letter
        for j in range(20):  # 20 examples per letter
            # Create a blank image
            img = np.ones((64, 64), dtype=np.uint8) * 255
            
            # Add noise
            noise = np.random.normal(0, 10, (64, 64)).astype(np.uint8)
            img = cv2.add(img, noise)
            
            # Draw letter
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 2
            text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
            
            # Position text in center
            x = (img.shape[1] - text_size[0]) // 2
            y = (img.shape[0] + text_size[1]) // 2
            
            cv2.putText(img, letter, (x, y), font, font_scale, (0, 0, 0), thickness)
            
            # Apply random rotation
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1)
            img = cv2.warpAffine(img, M, (64, 64))
            
            # Save image
            cv2.imwrite(os.path.join(letter_dir, f"{j}.png"), img)
    
    print(f"Created example dataset with {len(letters)} letters in {output_dir}")


def main(train=False):

    # Configuration
    data_dir = "hebrew_letters_dataset"
    img_height = 64
    img_width = 64
    batch_size = 32
    epochs = 50

    # Check if dataset exists, otherwise create example dataset
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Creating example dataset...")
        prepare_example_dataset(data_dir)

    # Initialize HebrewOCR
    ocr = HebrewOCR(data_dir, img_height, img_width)

    # Load dataset
    train_ds, val_ds, test_ds = ocr.load_dataset(batch_size=batch_size)

    if train:
        # Build and train model
        ocr.build_model()
        ocr.train(train_ds, val_ds, epochs=epochs)
    else:
        ocr.load_model("hebrew_ocr_model.h5")
    # Evaluate model
    ocr.evaluate(test_ds)

    if train:
        # Plot training history
        ocr.plot_training_history()
    
    # Visualize predictions
    ocr.visualize_predictions(num_samples=5)

    if train:
        # Save model
        ocr.save_model("hebrew_ocr_model.h5")
    
    # Test on a new document
    # Create a simple test document
    doc_img = np.ones((200, 600), dtype=np.uint8) * 255
    letters = ['א', 'ב', 'ג', 'ד', 'ה', 'ו']
    positions = [(50, 100), (150, 100), (250, 100), (350, 100), (450, 100), (550, 100)]
    
    for letter, pos in zip(letters, positions):
        cv2.putText(doc_img, letter, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Save test document
    cv2.imwrite("images/test_document.png", doc_img)
    
    # Process document
    result_img, text = ocr.process_document("test_document.png")
    cv2.imwrite("images/result_document.png", result_img)
    
    print(f"Recognized text: {text}")
    print("Results saved as 'result_document.png'")


def process(image_path):
    model = models.load_model('hebrew_ocr_model.h5')
    hebrew_letters = {
        0: 'ב', 1: 'כ', 2: 'פ', 3: 'ך', 4: 'ה', 5: 'ר', 6: 'ף', 7: 'ח', 8: 'מיקס',
        9: 'צ', 10: 'ק', 11: 'ט', 12: 'נ', 13: 'ת', 14: 'ג', 15: 'ן', 16: 'ם',
        17: 'ע', 18: 'י', 19: 'ד', 20: 'חצי-ק', 21: 'זבל', 22: 'ל', 23: 'א', 24: 'ז',
        25: 'ש', 26: 'זבל', 27: 'לא ידוע', 28: 'ו', 29: 'מ', 30: 'ץ', 31: 'ס'
    }
    img_height = img_width = 64
    min_size = 20
    padding = 5
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    result_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0], reverse=True)
    recognized_chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_size or h < min_size:
            continue
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(gray.shape[1], x + w + padding)
        y_end = min(gray.shape[0], y + h + padding)

        roi = gray[y_start:y_end, x_start:x_end]

        def switch_colors(image):
            return cv2.bitwise_not(image)
        roi = switch_colors(roi)
        # Resize to model input size
        roi_resized = cv2.resize(roi, (img_width, img_height))

        # Normalize
        roi_normalized = roi_resized.astype(np.float32) / 255.0

        # Add channel dimension and batch dimension
        roi_input = np.expand_dims(np.expand_dims(roi_normalized, axis=-1), axis=0)

        pred = model.predict(roi_input)[0]
        pred_class = np.argmax(pred)
        confidence = pred[pred_class]

        letter = hebrew_letters.get(pred_class, str(pred_class))
        recognized_chars.append(letter)

        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, f"{letter} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite("images/result_document.png", result_image)
    text = ''.join(recognized_chars)
    print(text)


if __name__ == "__main__":
    # main()
    process('parshiya.png')