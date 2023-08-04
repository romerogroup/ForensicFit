import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import joblib


def predict(image_front_path,image_back_path, project_dir='.'):
    batch_size = 5
    model_dir = os.path.join(project_dir,'models')

    load_path_back = os.path.join(model_dir,f'back_model_tensorflow')
    load_path_front = os.path.join(model_dir,f'front_model_tensorflow')

    if not os.path.exists(load_path_front):
        raise ValueError("Please download the models first. run python forensicfit.machine_learning.download_models()")
    
    #################################################################
    if not os.path.exists(image_front_path):
        raise ValueError("This image path does not exist")
    image_data=tf.io.read_file(image_front_path)
    # Decode the image data
    image_front = tf.image.decode_image(image_data)

    image_front=image_front[None,...]
    image_front=tf.image.resize(image_front, [1200,410])

    if not os.path.exists(image_back_path):
        raise ValueError("This image path does not exist")
    image_data=tf.io.read_file(image_back_path)
    image_back = tf.image.decode_image(image_data)
    
    image_back=image_back[None,...]
    image_back = tf.image.resize(image_back, [1200,410])

    #################################################################
    model_back = tf.keras.models.load_model(load_path_back)
    model_front = tf.keras.models.load_model(load_path_front)

    prediction_back=model_back.predict(image_back)
    prediction_front=model_front.predict(image_front)

    def print_result(output):
        if output >= 0.5:
            result='Fit'
            print(f"It's a {result}.")
        else:
            result='Non-Fit'
            print(f"It's a {result}.")
        return result
    print("Back prediction")
    result_back = print_result(prediction_back[0][0])
    print("Front prediction")
    result_front = print_result(prediction_front[0][0])

    # Combine model with the descision tree
    # dt = joblib.load(os.path.join(project_dir,'forensicfit','machine_learning','combinationDecisionTree.joblib'))
    # X = np.array([prediction_back[0][0],prediction_front[0][0]])
    # y_predicted = dt.predict(X)
    # print(y_predicted)
    return (prediction_back[0][0], result_back , 'back') , (prediction_front[0][0], result_front , 'front' )

if __name__ == '__main__':
    predict()