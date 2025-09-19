import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dataset Path and Parameters ---
dataset_dir = 'dataset'
img_width, img_height = 150, 150 
batch_size = 32
# --- CHANGE 1: Increased epochs for better convergence ---
epochs = 30 

# --- Check if dataset exists ---
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset folder named '{dataset_dir}' not found.")
    print("Please create the dataset folder with 'good' and 'bad' subdirectories.")
else:
    # --- Preprocess and augment image data ---
    # More augmentation for a robust model
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2, # Randomly shift images vertically
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% of data for validation
    )

    # Separate generator for validation data - no augmentation, just rescaling
    validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # --- Training Data Generator ---
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    # --- Validation Data Generator ---
    validation_generator = validation_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False # --- CHANGE 2: CRUCIAL for correct evaluation ---
    )

    # --- Build a Deeper and More Robust CNN Model ---
    # --- CHANGE 3: Improved Model Architecture ---
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # --- Compile the Model with a controlled learning rate ---
    # --- CHANGE 4: Lower learning rate for more stable training ---
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Print model architecture

    print("\nStarting model training...")
    
    # --- Train the Model ---
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )

    # --- Save the Trained Model ---
    model.save('image_quality_model.keras')
    print("\nModel training complete and saved as 'image_quality_model.keras'.")

    # --- Plot Training History ---
    # --- CHANGE 5: Visualize training to diagnose issues ---
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # --- Correct Evaluation ---
    print("\n--- Model Evaluation ---")
    # Get the ground truth labels for the validation set
    validation_labels = validation_generator.classes

    # Predict on the entire validation set
    # The number of steps ensures we cover all samples
    predictions = model.predict(validation_generator, steps=(validation_generator.samples // batch_size) + 1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Important: Slice both arrays to the same size to avoid errors
    # if the last batch was not full.
    num_samples = len(validation_generator.filenames)
    validation_labels = validation_labels[:num_samples]
    predicted_classes = predicted_classes[:num_samples]

    # Generate and print the confusion matrix
    cm = confusion_matrix(validation_labels, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)

    # Generate and print the classification report
    print("\nClassification Report:")
    # Use the generator's class indices to get the correct label names
    target_names = list(validation_generator.class_indices.keys())
    print(classification_report(validation_labels, predicted_classes, target_names=target_names))

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
