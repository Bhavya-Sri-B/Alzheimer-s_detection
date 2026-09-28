"""
Alzheimer's MRI Classifier - Model Testing Script
Tests the trained EfficientNetB5 model on the test dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Settings
IMG_SIZE = (176, 208)
BATCH_SIZE = 32
MODEL_PATH = r"g:\Alzemer's\Alzheimer_s\Models\efficientnetb5_custom_new.h5"
TEST_DIR = r"g:\Alzemer's\Alzheimer_s\Combined Dataset\test"

# Class names
class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

print("=" * 60)
print("ALZHEIMER'S MRI CLASSIFIER - MODEL TESTING")
print("=" * 60)

# Load model
print("\n[1/4] Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# Load test dataset
print("\n[2/4] Loading test dataset...")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    seed=54,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
print(f"✓ Test dataset loaded from: {TEST_DIR}")
print(f"  Classes found: {test_ds.class_names}")

# One-hot encode labels
def one_hot_label(image, label):
    label = tf.one_hot(label, len(class_names))
    return image, label

test_ds_prefetch = test_ds.map(one_hot_label).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Evaluate model
print("\n[3/4] Evaluating model on test set...")
results = model.evaluate(test_ds_prefetch, verbose=1)
print(f"\n✓ Test Results:")
print(f"  Loss: {results[0]:.4f}")
for i, metric_name in enumerate(model.metrics_names[1:], 1):
    print(f"  {metric_name}: {results[i]:.4f}")

# Predictions
print("\n[4/4] Generating predictions...")
y_pred = model.predict(test_ds_prefetch, verbose=1)
predicted_categories = tf.argmax(y_pred, axis=1).numpy()

# Get true labels
true_categories = tf.concat([y for x, y in test_ds_prefetch], axis=0)
true_categories = tf.argmax(true_categories, axis=1).numpy()

# Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(true_categories, predicted_categories, target_names=class_names))

# Confusion Matrix
print("\n[Generating Confusion Matrix...]")
cm = confusion_matrix(true_categories, predicted_categories)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp1.plot(ax=axes[0], cmap=plt.cm.Blues, xticks_rotation=45)
axes[0].set_title('Confusion Matrix (Counts)')

# Normalized confusion matrix
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp2.plot(ax=axes[1], cmap=plt.cm.Blues, xticks_rotation=45)
axes[1].set_title('Confusion Matrix (Normalized)')

plt.tight_layout()
plt.savefig(r"g:\Alzemer's\Alzheimer_s\confusion_matrix.png", dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved to: confusion_matrix.png")
plt.show()

# Sample predictions visualization
print("\n[Showing sample predictions...]")
plt.figure(figsize=(15, 10))
for images, labels in test_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].numpy().astype("uint8")
        plt.imshow(img)
        
        # Predict
        pred = model.predict(tf.expand_dims(images[i], 0), verbose=0)
        pred_class = class_names[np.argmax(pred[0])]
        true_class = test_ds.class_names[labels[i]]
        confidence = np.max(pred[0]) * 100
        
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)", 
                  color=color, fontsize=9)
        plt.axis("off")

plt.tight_layout()
plt.savefig(r"g:\Alzemer's\Alzheimer_s\sample_predictions.png", dpi=150, bbox_inches='tight')
print("✓ Sample predictions saved to: sample_predictions.png")
plt.show()

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
