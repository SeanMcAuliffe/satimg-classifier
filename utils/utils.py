import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import json
import seaborn as sns
from tensorflow import keras
from keras import backend as K
import gc

VIS_DIR_PATH = os.path.join("..", "vis")
RESULTS_DIR = os.path.join("..", "data", "results")
COORD_FILEPATH= os.path.join("..", "data", "labels", "coord.csv")


def save_model(model, name):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    RESULTS_PATH = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    model_path = os.path.join(RESULTS_PATH, f'{name}.h5')
    model.save(model_path)
    print(f'Model saved to {model_path}')


def log_experiment(history, name, description, x_val, y_val, Y_train, tr_names, val_names, reg: bool = False):
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    RESULTS_PATH = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    model = keras.models.load_model(os.path.join(RESULTS_PATH, f'{name}.h5'))

    with open(os.path.join(RESULTS_PATH, "model_details.txt"), 'w') as f:
        f.write(description + '\n\n')
        for i, layer in enumerate(model.layers):
            # Write the layer index, layer name, and layer configuration to the text file
            f.write(f'Layer {i}: {layer.name}\n')
            f.write(f'Configuration:\n')
            f.write(str(layer.get_config()) + '\n\n')

    history_path = os.path.join(RESULTS_PATH, f'{name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f'Training history saved to {history_path}')

    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    performance_path = os.path.join(RESULTS_PATH, f'{name}_performance.txt')
    with open(performance_path, 'w') as f:
        f.write(f'Loss: {loss}\n')
        f.write(f'Accuracy: {accuracy}\n')
    print(f'Model performance saved to {performance_path}')

    # Get the model's predictions for the validation set
    y_pred = model.predict(x_val)

    del model
    gc.collect()
    K.clear_session()
    
    if reg:
        plot_accuracy(history, name, name + "_acc.png", RESULTS_PATH)
        return

    # Convert the predictions to binary labels
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    y_true_classes = y_val.flatten()

    del y_pred
    gc.collect()

    # Compare the predicted and true labels to find the correct and incorrect examples
    correct_indices = np.where(y_pred_classes == y_true_classes)[0]
    incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]

    del y_pred_classes
    del y_true_classes
    gc.collect()

    coords = {}
    with open(COORD_FILEPATH, 'r') as f:
        for line in f.readlines():
            im_name, lat, lng = line.rstrip().split(',')
            coords[im_name] = (float(lat[1:]), float(lng[:-1]))

    tr_pos_indices = np.where(Y_train == 1)[0]
    tr_neg_indices = np.where(Y_train == 0)[0]

    val_pos_inidices = np.where(y_val == 1)[0]
    val_neg_inidices = np.where(y_val == 0)[0]

    tr_pos_names = [tr_names[i] for i in tr_pos_indices]
    tr_neg_names = [tr_names[i] for i in tr_neg_indices]

    del tr_names
    del tr_pos_indices
    del tr_neg_indices
    gc.collect()

    val_pos_names = [val_names[i] for i in val_pos_inidices]
    val_neg_names = [val_names[i] for i in val_neg_inidices]


    del val_pos_inidices
    del val_neg_inidices
    gc.collect()

    val_pos_coords = [coords[name] for name in val_pos_names]
    val_neg_coords = [coords[name] for name in val_neg_names]

    tr_pos_coords = [coords[name] for name in tr_pos_names]
    tr_neg_coords = [coords[name] for name in tr_neg_names]

    val_correct_pos_names = [val_names[i] for i in correct_indices if y_val[i] == 1]
    val_correct_neg_names = [val_names[i] for i in correct_indices if y_val[i] == 0]

    val_true_pos_coords = [coords[name] for name in val_correct_pos_names]
    val_true_neg_coords = [coords[name] for name in val_correct_neg_names]

    
    val_incorrect_pos_names = [val_names[i] for i in incorrect_indices if y_val[i] == 1]
    val_incorrect_neg_names = [val_names[i] for i in incorrect_indices if y_val[i] == 0]

    val_false_pos_coords = [coords[name] for name in val_incorrect_pos_names]
    val_false_neg_coords = [coords[name] for name in val_incorrect_neg_names]

    total = len(val_pos_names) + len(val_neg_names)

    del val_neg_names
    del val_pos_names
    gc.collect()

    with open(os.path.join(RESULTS_PATH, f'{name}_record.txt'), 'w') as f:
        f.write(f"Total: {total} -- False Positives {len(val_incorrect_pos_names)} -- False Negatives {len(val_incorrect_neg_names)}\n\n")
        f.write("True Positives:\n")
        for pos_name in val_correct_pos_names:
            f.write(f'{pos_name}\n')
        f.write("True Negatives\n")
        for neg_name in val_correct_neg_names:
            f.write(f'{neg_name}\n')
        f.write("False Positives:\n")
        for pos_name in val_incorrect_pos_names:
            f.write(f'{pos_name}\n')
        f.write("False Negatives:\n")
        for neg_name in val_incorrect_neg_names:
            f.write(f'{neg_name}\n')

    del val_correct_pos_names
    del val_correct_neg_names
    gc.collect()

    
    del val_names
    del val_incorrect_neg_names
    del val_incorrect_pos_names
    del coords
    del correct_indices
    del incorrect_indices
    gc.collect()

    # Plot and save accuracy and val accuracy.
    plot_accuracy(history, name, name + "_acc.png", RESULTS_PATH)

    del history
    gc.collect()
    
    # # Plot and save training set pos/neg.
    plot_on_world_map_pos_neg(tr_pos_coords,
                              tr_neg_coords,
                              title = f"{name}: Training Set",
                              path=RESULTS_PATH,
                              xlabel = "lat",
                              ylabel = "lon",
                              filename = f"{name}_train.png")
    
    # # Plot and save testing set pos/neg.
    plot_on_world_map_pos_neg(val_pos_coords,
                              val_neg_coords,
                              title = f"{name}: Testing Set",
                              path=RESULTS_PATH,
                              xlabel = "lat",
                              ylabel = "lon",
                              filename = f"{name}_test.png")

    # # Plot and save correct predictions pos/neg.
    plot_on_world_map_pos_neg(val_true_pos_coords,
                              val_true_neg_coords,
                              title = f"{name}: True Pos/Neg",
                              path=RESULTS_PATH,
                              xlabel = "lat",
                              ylabel = "lon",
                              filename = f"{name}_true.png")

    # # Plot and save incorrect predictions pos/neg.
    plot_on_world_map_pos_neg(val_false_pos_coords,
                              val_false_neg_coords,
                              title = f"{name}: False Pos/Neg",
                              path=RESULTS_PATH,
                              xlabel = "lat",
                              ylabel = "lon",
                              filename = f"{name}_false.png")


class ImageBucket():
    
    def __init__(self, location_code, example_list, brightness):
        self.bright_dict = brightness
        self.l_code = location_code
        self.e_list = example_list
        self.next_index = 0
        self.cardinality = len(self.e_list)
        self.sort_by_desireability()

    def sort_by_desireability(self):
        self.e_list.sort(key=lambda x: self.bright_dict[x])

    def get_next_image(self):
        e = self.e_list[self.next_index]
        self.next_index = (self.next_index + 1) % self.cardinality
        return e


def get_bounding_box(md_filepath):
    # From metadata file, obtain coords of bounding box of image.
    bounding_box = {}
    with open(md_filepath) as md_file:
        # Scan lines only until all corner coords are found.
        for line in md_file:
            key, val  = line.partition("=")[::2]
            key = key.strip()
            val = val.strip()
            if key.endswith("LAT_PRODUCT") or key.endswith("LON_PRODUCT"):
                bounding_box[key] = float(val)
            if len(bounding_box) == 8:
                break
    return bounding_box


def analyze_bounding_box(bounding_box):
    # Obtain vectors of bounding box edges.
    ul = [bounding_box["CORNER_UL_LAT_PRODUCT"], bounding_box["CORNER_UL_LON_PRODUCT"]]
    ur = [bounding_box["CORNER_UR_LAT_PRODUCT"], bounding_box["CORNER_UR_LON_PRODUCT"]]
    ll = [bounding_box["CORNER_LL_LAT_PRODUCT"], bounding_box["CORNER_LL_LON_PRODUCT"]]
    lr = [bounding_box["CORNER_LR_LAT_PRODUCT"], bounding_box["CORNER_LR_LON_PRODUCT"]]
    edges = [] # Has form [edge start point, edge direction]
    edges.append([ul, [(ur[0] - ul[0]), (ur[1] - ul[1])]])
    edges.append([ur, [(lr[0] - ur[0]), (lr[1] - ur[1])]])
    edges.append([lr, [(ll[0] - lr[0]), (ll[1] - lr[1])]])
    edges.append([ll, [(ul[0] - ll[0]), (ul[1] - ll[1])]])
    return ul, ur, ll, lr, edges
    

def plot_on_world_map_pos_neg(pos, neg, title, xlabel, ylabel, filename, path):
    sns.set_style("white")
    bg_image = mpimg.imread(os.path.join(VIS_DIR_PATH, "world-map.png"))

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor((1, 1, 1, 0))

    x_vals = [point[1] for point in pos]
    y_vals = [point[0] for point in pos]
    ax.plot(x_vals, y_vals, marker=".", markersize=0.5, lw=0.01, linestyle="None", color="blue", label="pos")

    x_vals = [point[1] for point in neg]
    y_vals = [point[0] for point in neg]
    ax.plot(x_vals, y_vals, marker=".", markersize=0.5, lw=0.01, linestyle="None", color="red", label="neg")

    ax.imshow(bg_image, extent=[-180.0, 180.0, -90.0, 90.0])

    plt.xlim([-180.0, 180.0])
    plt.ylim([-90.0, 90.0])
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(path, filename), dpi=1200)

def plot_accuracy(history, name, filename, path):
    #sns.set_style("darkgrid")
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.title(f"{name}: Accuracy")
    #Save the plot to disc
    plt.savefig(os.path.join(path, filename))
