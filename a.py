import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define paths and image dimensions
train_dir = r'F:\ml model\Datasets\Train'
validation_dir = r'F:\ml model\Datasets\Validation'
IMG_HEIGHT = 48
IMG_WIDTH = 48

print("=== DATASET CHECK ===")

# Check directory structure
def check_directory_structure(directory_path, directory_name):
    print(f"\n--- Checking {directory_name} directory ---")
    
    if not os.path.exists(directory_path):
        print(f"❌ ERROR: Directory does not exist: {directory_path}")
        return []
    
    contents = os.listdir(directory_path)
    print(f"Found {len(contents)} items in {directory_name}:")
    
    classes = []
    for item in contents:
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            num_images = len([f for f in os.listdir(item_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            classes.append(item)
            print(f"  📁 {item}: {num_images} images")
        else:
            print(f"  📄 {item} (file)")
    
    return classes

# Check both directories
train_classes = check_directory_structure(train_dir, "Training")
validation_classes = check_directory_structure(validation_dir, "Validation")

# Check if directories have the same classes
if train_classes and validation_classes:
    if set(train_classes) == set(validation_classes):
        print(f"\n✅ Both directories have the same classes: {train_classes}")
    else:
        print(f"\n⚠️  Warning: Training and validation directories have different classes!")
        print(f"Training classes: {train_classes}")
        print(f"Validation classes: {validation_classes}")

print(f"\n=== SETUP DATA GENERATORS ===")

# Create an ImageDataGenerator that rescales, resizes, and converts to grayscale
datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the training data
print("\nLoading training data...")
train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

# Load and prepare the validation data
print("Loading validation data...")
validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

# Display generator information
print(f"\n✅ Data generators created successfully!")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Number of classes: {len(train_generator.class_indices)}")
print(f"Class mapping: {train_generator.class_indices}")
print(f"Batch size: {train_generator.batch_size}")
print(f"Input shape: {train_generator.image_shape}")

print(f"\n=== BUILDING MODEL ===")

# Get the actual number of classes from the data
num_classes = len(train_generator.class_indices)

print(f"Building model for {num_classes} classes...")

model = Sequential([
    # 1st Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(2, 2),
    
    # 2nd Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Flatten the results to feed into a dense layer
    Flatten(),
    
    # Dense Layer
    Dense(1024, activation='relu'),
    Dropout(0.5),
    
    # Output Layer - dynamically set to match number of classes
    Dense(num_classes, activation='softmax')
])

# Print a summary of the model
print("\nModel Summary:")
model.summary()

print(f"\n=== COMPILING MODEL ===")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(f"\n=== STARTING TRAINING ===")

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Starting training for 30 epochs...")

# Train the model
history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        verbose=1)

print(f"\n=== SAVING MODEL ===")
# Save the model to a file
model.save('facial_expression_model.h5')
print("✅ Model saved successfully as 'facial_expression_model.h5'")

print(f"\n=== TRAINING COMPLETE ===")
print("Your model has been trained and saved!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Optional: Plot training history (if you want to visualize)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("✅ Training history plot saved as 'training_history.png'")
    
except ImportError:
    print("ℹ️  Install matplotlib to see training plots: pip install matplotlib")