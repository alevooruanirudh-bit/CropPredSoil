from __future__ import annotations
from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Alluvial", "Black", "Clay", "Red"]


def train(soil_dir: str | Path, models_dir: str | Path = "models", epochs: int = 30) -> dict:
    soil_dir = Path(soil_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    ds = tf.keras.utils.image_dataset_from_directory(
        soil_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="both",
    )
    train_ds, val_ds = ds
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    base.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(models_dir / "soil_mobilenetv2.keras", monitor="val_accuracy", save_best_only=True),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    metrics = {
        "train_images": int(sum(1 for _ in train_ds.unbatch())),
        "val_images": int(sum(1 for _ in val_ds.unbatch())),
        "final_train_accuracy": float(hist.history["accuracy"][-1]),
        "final_val_accuracy": float(hist.history["val_accuracy"][-1]),
        "best_val_accuracy": float(max(hist.history["val_accuracy"])),
        "class_names": CLASS_NAMES,
    }
    with open(models_dir / "soil_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def load(models_dir: str | Path = "models"):
    return tf.keras.models.load_model(Path(models_dir) / "soil_mobilenetv2.keras")


def predict(model, image_path: str | Path) -> dict:
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(tf.argmax(probs))
    return {"soil_type": CLASS_NAMES[idx], "probability": float(probs[idx]), "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}}
