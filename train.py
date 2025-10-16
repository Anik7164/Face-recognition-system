from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define paths and image dimensions
train_dir = 'F:\ml model\Datasets\Train'
validation_dir = 'F:\ml model\Datasets\Validation'
IMG_HEIGHT = 48
IMG_WIDTH = 48

# Create an ImageDataGenerator that rescales, resizes, and converts to grayscale
datagen = ImageDataGenerator(rescale=1./255) # Rescales pixel values from 0-255 to 0-1

# Load and prepare the training data
train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),  # <-- This resizes all images
        color_mode='grayscale',              # <-- This converts all images to grayscale
        batch_size=64,
        class_mode='categorical')

# Load and prepare the validation data
validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),  # <-- This resizes all images
        color_mode='grayscale',              # <-- This converts all images to grayscale
        batch_size=64,
        class_mode='categorical')

# FIXED: Removed extra space before this line
model = Sequential([
    # 1st Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
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
    
    # Output Layer
    # The number of neurons must match the number of classes (emotions)
    Dense(8, activation='softmax') 
])

# Print a summary of the model
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# FIXED: Proper indentation for training code
# Train the model
history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100, # You can adjust this number
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size)

# Save the model to a file
model.save('facial_expression_model.h5')
print("Model saved successfully!")