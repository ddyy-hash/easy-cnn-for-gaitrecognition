"""Model definition for the compact 3D CNN gait baseline."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, Input, MaxPooling3D


def build_3d_cnn(input_shape: tuple[int, int, int, int], num_classes: int) -> Sequential:
    """Build the original compact 3D CNN topology with modern Keras APIs.

    Args:
        input_shape: Video tensor shape as ``(frames, height, width, channels)``.
        num_classes: Number of identity or gait classes.

    Returns:
        A compiled-ready Keras ``Sequential`` model.
    """
    model = Sequential(name="easy_gait_3dcnn")
    model.add(Input(shape=input_shape, name="video_clip"))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding="same", name="conv3d_32_a"))
    model.add(Activation("relu", name="relu_32_a"))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding="same", name="conv3d_32_b"))
    model.add(Activation("softmax", name="softmax_32_b"))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same", name="pool_32"))
    model.add(Dropout(0.25, name="dropout_32"))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding="same", name="conv3d_64_a"))
    model.add(Activation("relu", name="relu_64_a"))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding="same", name="conv3d_64_b"))
    model.add(Activation("softmax", name="softmax_64_b"))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same", name="pool_64"))
    model.add(Dropout(0.25, name="dropout_64"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(512, activation="sigmoid", name="embedding_dense"))
    model.add(Dropout(0.5, name="embedding_dropout"))
    model.add(Dense(num_classes, activation="softmax", name="class_prediction"))

    return model

