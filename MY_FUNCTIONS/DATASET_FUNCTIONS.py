import math
import os

import matplotlib.pyplot as plt
import tensorflow as tf


def parse_label_from_filename(filename):
    # Извлекаем имя файла без пути
    basename = tf.strings.split(filename, os.sep)[-1]
    # Убираем расширение и превращаем в строку
    label_str = tf.strings.regex_replace(basename, r'\.[^.]+$', '')
    # Преобразуем в float32 для получения числовой метки
    label = tf.strings.to_number(label_str, out_type=tf.float32)
    return label


def load_image_with_label(file_path):
    # Загрузим изображение
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Приводим изображение к float32
    label = parse_label_from_filename(file_path)
    return image, label


def create_dataset_from_directory(directory_path, batch_size=1):
    # Получаем все пути к файлам
    file_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if
                  fname.endswith('.jpg') or fname.endswith('.png')]

    # Создаем Dataset из путей
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_image_with_label, num_parallel_calls=tf.data.AUTOTUNE)
    # Убедитесь, что размер батча не превышает размер датасета
    dataset = dataset.batch(batch_size)
    return dataset


def show_all_images_in_grid(dataset, num_cols=5):
    # Получаем количество изображений в датасете
    dataset_size = sum(1 for _ in dataset)
    num_rows = math.ceil(dataset_size / num_cols)

    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for i, (image, label) in enumerate(dataset):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        # Если изображение одноканальное, указываем cmap='gray'
        if image.shape[-1] == 1:
            plt.imshow(image.numpy().squeeze(), cmap='gray')
        else:
            plt.imshow(image.numpy())
        plt.title(f"Label: {label.numpy()}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_predictions(dataset, model, num_images=5):
    plt.figure(figsize=(15, 6))

    for i, (image, true_label) in enumerate(dataset.take(num_images)):
        predicted_label = model.predict(image)

        plt.subplot(2, num_images, i + 1)
        plt.imshow(image[0, :, :, 0], cmap='gray')  # Отображение изображения
        plt.title(f'True: {true_label.numpy()[0]:.2f}\nPredicted: {predicted_label[0][0]:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_prediction_vs_true(dataset, model):
    true_labels = []
    predicted_labels = []

    for image, true_label in dataset:
        predicted_label = model.predict(image)
        true_labels.append(true_label.numpy()[0])
        predicted_labels.append(predicted_label[0][0])

    plt.figure(figsize=(8, 8))
    plt.scatter(true_labels, predicted_labels, alpha=0.6)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')  # Линия y=x
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.grid()
    plt.show()
