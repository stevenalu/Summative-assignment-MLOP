import tensorflow as tf
from preprocessing import build_dataset, train_dir, test_dir, model_path


def load_trained_model(path=model_path):
    return tf.keras.models.load_model(path)


def retrain(epochs=5, path=model_path):
    model = load_trained_model(path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    train_ds = build_dataset(train_dir, shuffle=True)
    val_ds = build_dataset(test_dir, shuffle=False)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(path)
    return history.history