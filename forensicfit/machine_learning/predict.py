import tensorflow as tf
import os
import matplotlib.pyplot as plt

from forensicfit.utils import ROOT



def predict():
    batch_size = 5
    model_dir = os.path.join(ROOT,'models')

    model_type = 'back'
    load_path = os.path.join(model_dir,f'{model_type}_model_tensorflow')

    processed_dir = os.path.join(ROOT, "data", "processed", "normal_split", "match_nonmatch_ratio_0.3")

    train_dir = os.path.join(processed_dir, model_type, "train")
    test_dir = os.path.join(processed_dir, model_type, "test")

    tf.random.set_seed(0)
    ######################################################
    ######################################################
    ######################################################
    ######################################################

    def random_flip(image, label):
        # image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
        return image, label

    train_dataset = tf.keras.utils.image_dataset_from_directory(directory=train_dir,color_mode='grayscale',image_size=(1200,410), batch_size=batch_size)
    test_dataset = tf.keras.utils.image_dataset_from_directory(directory=test_dir,color_mode='grayscale',image_size=(1200,410), batch_size=batch_size)

    n_train = len(train_dataset)*batch_size
    n_test = len(test_dataset)*batch_size

    # Apply random vertical and horizontal flip augmentation
    train_dataset = train_dataset.map(random_flip)

    model = tf.keras.models.load_model(load_path)

    test_loss, test_acc,tp,fp,tn,fn,auc = model.evaluate(test_dataset)
    print(f'Test accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    predict()