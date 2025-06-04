import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


np.random.seed(42)


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def net() -> keras.models.Model:
    def conv2d_block(filters: int, kernel_size: tuple[int], dropout: float):
        return (
            Conv2D(filters, kernel_size, kernel_initializer='he_uniform', padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(filters, kernel_size, kernel_initializer='he_uniform', padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(dropout)
        )

    def fc(linear_units: int, dropout: float, num_classes: int):
        return (
            Flatten(),
            Dense(linear_units, kernel_initializer='he_uniform', activation='relu'),
            BatchNormalization(),
            Dropout(dropout),
            Dense(num_classes, activation='softmax')
        )

    model = Sequential((
        Input((32, 32, 3)),
        *conv2d_block(32, (3, 3), 0.2),
        *conv2d_block(64, (3, 3), 0.3),
        *conv2d_block(128, (3, 3), 0.4),
        *fc(128, 0.5, 10)
    ))

    return model


def train_loop(model, x_train, y_train, x_test, y_test) -> keras.callbacks.History:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy()
        ]
    )

    return model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))


def plot_confusion_matrix(y_true, y_pred, ax):
    labels = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_xlabel('Predicted label', fontsize=18)
    ax.set_ylabel('True label', fontsize=18)

    ax.set_title("Confusion Matrix", fontsize=20)


def plot_training_history(history, ax):
    for metric, values in history.history.items():
        ax.plot(values, label=metric)

    ax.legend(fontsize=20)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Metric Value', fontsize=18)

    ax.set_title('Training history', fontsize=20)
    ax.grid(True)


def main():
    x_train, y_train, x_test, y_test = load_cifar10()

    model = net()

    history = train_loop(model, x_train, y_train, x_test, y_test)
    model.save('model.keras')

    model = keras.models.load_model('model.keras')

    predictions = np.argmax(model.predict(x_test), axis=1)

    keras.utils.plot_model(model, to_file='../assets/model.png', show_shapes=True, show_layer_names=True)

    fig, axs = plt.subplots(1, 2, figsize=(43.20, 21.60))
    left_ax, right_ax = axs

    plot_confusion_matrix(y_test.flatten(), predictions, left_ax)
    plot_training_history(history, right_ax)

    accuracy = accuracy_score(y_test.flatten(), predictions)
    fig.suptitle(f'CIFAR-10 Convolutional Neural Network (Test Accuracy: {accuracy:.3f})', fontsize=24)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig('cifar10-training-report.png')


if __name__ == '__main__':
    main()