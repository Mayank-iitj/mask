"""Face mask classifier CLI

This script provides three modes:
        - train: train a simple CNN from folders `train/` and `test/` (configurable)
        - test-single: run a single image prediction
        - live: run webcam-based live detection

It uses tensorflow.keras for compatibility with most deployment targets.
"""

from __future__ import annotations

import argparse
import os
import sys
import datetime
import logging
from typing import Optional

import numpy as np
import cv2

try:
        from tensorflow import keras
        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
except Exception:  # fallback to keras if tensorflow not available
        import keras
        from keras import backend as K
        from keras.models import Sequential, load_model
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
        from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

LOG = logging.getLogger(__name__)


def build_model(input_shape=(150, 150, 3)) -> keras.Model:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model


def train_model(train_dir: str, val_dir: str, model_out: str, epochs: int = 10, batch_size: int = 16):
        LOG.info("Starting training: train=%s val=%s epochs=%s batch_size=%s", train_dir, val_dir, epochs, batch_size)
        if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(val_dir):
                raise FileNotFoundError(f"Validation directory not found: {val_dir}")

        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        training_set = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_size,
                                                                                                         class_mode="binary")
        validation_set = test_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=batch_size,
                                                                                                          class_mode="binary")

        model = build_model()
        history = model.fit(training_set, epochs=epochs, validation_data=validation_set)
        model.save(model_out)
        LOG.info("Training completed and model saved to %s", model_out)
        return history


def predict_single(model_path: str, image_path: str) -> float:
        if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

        model = load_model(model_path)
        img = load_img(image_path, target_size=(150, 150))
        arr = img_to_array(img)
        arr = np.expand_dims(arr.astype("float32") / 255.0, axis=0)
        prob = float(model.predict(arr)[0][0])
        LOG.info("Prediction for %s => probability=%f", image_path, prob)
        return prob


def run_live(model_path: str, cascade_path: str, cam_index: int = 0, threshold: float = 0.5):
        if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(cascade_path):
                raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

        model = load_model(model_path)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
                raise RuntimeError("Unable to open camera")
        face_cascade = cv2.CascadeClassifier(cascade_path)

        try:
                while True:
                        ret, frame = cap.read()
                        if not ret:
                                LOG.warning("Camera frame not available")
                                break

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                        for (x, y, w, h) in faces:
                                face_img = frame[y:y + h, x:x + w]
                                face_resized = cv2.resize(face_img, (150, 150))
                                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                                inp = np.expand_dims(face_rgb.astype("float32") / 255.0, axis=0)
                                prob = float(model.predict(inp)[0][0])
                                label = "NO MASK" if prob > threshold else "MASK"
                                color = (0, 0, 255) if prob > threshold else (0, 255, 0)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                                cv2.putText(frame, label, ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                        datet = str(datetime.datetime.now())
                        cv2.putText(frame, datet, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.imshow("Face Mask Detector", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
        finally:
                cap.release()
                cv2.destroyAllWindows()


def parse_args(argv: Optional[list[str]] = None):
        p = argparse.ArgumentParser(description="Face Mask Detection CLI")
        sub = p.add_subparsers(dest="mode", required=True)

        t = sub.add_parser("train", help="Train a new model")
        t.add_argument("--train-dir", required=True, help="Path to training data (folder with class subfolders)")
        t.add_argument("--val-dir", required=True, help="Path to validation data")
        t.add_argument("--out", default="mymodel.h5", help="Output model path")
        t.add_argument("--epochs", type=int, default=10)
        t.add_argument("--batch-size", type=int, default=16)

        ts = sub.add_parser("test-single", help="Run single image prediction")
        ts.add_argument("--model", required=True, help="Model path")
        ts.add_argument("--image", required=True, help="Image path to classify")

        l = sub.add_parser("live", help="Run live webcam detection")
        l.add_argument("--model", required=True, help="Model path")
        l.add_argument("--cascade", default="haarcascade_frontalface_default.xml", help="Haar cascade path")
        l.add_argument("--cam", type=int, default=0, help="Camera index")
        l.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")

        return p.parse_args(argv)


def main(argv: Optional[list[str]] = None):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        args = parse_args(argv)

        if args.mode == "train":
                train_model(args.train_dir, args.val_dir, args.out, epochs=args.epochs, batch_size=args.batch_size)
        elif args.mode == "test-single":
                prob = predict_single(args.model, args.image)
                print(f"Prediction probability (no-mask): {prob:.4f}")
                sys.exit(0 if prob <= 0.5 else 1)
        elif args.mode == "live":
                run_live(args.model, args.cascade, cam_index=args.cam, threshold=args.threshold)


if __name__ == "__main__":
        main()
