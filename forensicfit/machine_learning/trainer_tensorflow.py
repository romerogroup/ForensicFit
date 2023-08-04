import tensorflow as tf
import os
import matplotlib.pyplot as plt


from forensicfit.utils import ROOT


def train():
    model_dir = os.path.join(ROOT,'models')

    # Hyperparameters
    num_epochs = 100
    # learning_rate = 1e-4
    batch_size = 5
    early_stopping = True
    patience = 10

    model_type = 'back'

    save_model = True 
    save_path = os.path.join(model_dir,f'{model_type}_model_tensorflow')

    processed_dir = os.path.join(ROOT, "data", "processed", "normal_split", "match_nonmatch_ratio_0.3")
    # processed_dir = os.path.join(PROJECT_DIR, "data", "processed", "cross_validation", "match_nonmatch_ratio_0.3", "0")

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

    model = tf.keras.Sequential(
                        [
                            tf.keras.layers.InputLayer(input_shape=(1200,410,1)),
                            tf.keras.layers.Conv2D(filters = 32, kernel_size=3, strides=(1, 1), padding="same", activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                            
                            tf.keras.layers.Conv2D(filters= 64, kernel_size=3, strides=(1, 1), padding="same",activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                            
                            tf.keras.layers.Conv2D(filters= 128, kernel_size=3, strides=(1, 1), padding="same",activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                            
                            tf.keras.layers.Conv2D(filters= 256, kernel_size=3, strides=(1, 1), padding="same",activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                            
                            tf.keras.layers.Conv2D(filters= 512, kernel_size=3, strides=(1, 1), padding="same",activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),

                            tf.keras.layers.Conv2D(filters= 1024, kernel_size=3, strides=(1, 1), padding="same",activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
                        
                            tf.keras.layers.Flatten(),

                            tf.keras.layers.Dropout(0.5),
                            tf.keras.layers.Dense(500, activation='relu'),
                            tf.keras.layers.Dropout(0.5),
                            tf.keras.layers.Dense(100, activation='relu'),
                            tf.keras.layers.Dense(1, activation = 'sigmoid'),
                        ])


    metrics = ['accuracy',
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.AUC(name='auc'),
                ]

    loss=tf.keras.losses.BinaryCrossentropy()

    num_train_steps = (n_train/batch_size)*num_epochs
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate = 0.0001,end_learning_rate = 0.00001, decay_steps = num_train_steps,power=2.0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode="auto",
            patience=patience,
            restore_best_weights=True
            # start_from_epoch=0,
        )
        callbacks = [early_stopping_callback]
    else:
        callbacks = None


    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    # Train the model
    history = model.fit(train_dataset, 
                        validation_data=test_dataset,
                        epochs=num_epochs, 
                        callbacks=callbacks)

    test_loss, test_acc,tp,fp,tn,fn,auc = model.evaluate(test_dataset)
    print(f'Test accuracy: {test_acc:.4f}')

    if save_model:
        model.save(save_path)

if __name__ == '__main__':
    train()