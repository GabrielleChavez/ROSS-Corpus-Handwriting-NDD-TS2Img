# Copyright (C) 2026 Gabrielle Chavez
# Johns Hopkins University 

import multiprocessing
import os, random, re, io
from typing import Dict, List, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset 
from transformers import AutoImageProcessor, ResNetForImageClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


######################## CONSTANTS #######################
# Use a class or a dictionary for constants to keep them organized
class Config:
    TEST_TRANS = True
    SHOW_DATA_SPLITS = False 
    NLS_DATA = r"PD|CTL"
    LABEL_1 = "PD"
    LABEL_2 = "CTL"
    RUN_COPRA = True

    RANDOM_STATE = 42
    N_SPLITS = 5
    BATCH_SIZE = 16
    MODEL_NAME = "microsoft/resnet-50"
    PATIENCE = 10
    NUM_EPOCHS = 50
    BASE_PATH = "/../../projects/NLS_ADPIE/data/NLS/handwriting/clean/" 
    DATA_PATH = "/projects/NLS_ADPIE/data/"


######################### IMAGE PROCESSING FUNCTIONS #########################
random.seed(42)
np.random.seed(42)

def loadImages(d, spiralType, directory = "Data/"):
    # ResNet50 expects ImageNet normalization

    count = 1
    data = {} # dictionary key = patient and value = image

    # directory = "/projects/NLS_ADPIE/data/" + d
    # directory = "Data/" + d
    directory += d
    S_TYPE = Config.LABEL_1 if Config.LABEL_1 in spiralType else Config.LABEL_2
    for photo in os.listdir(directory):
        filepath = os.path.join(directory, photo)
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = setLuminosity(img)
            img = img.resize((224, 224))
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            data[f'{spiralType}_{count}'] = {"image": img, "label": S_TYPE}
            count += 1

    return data

def setLuminosity(img, target_lum=275):
    """
    Adjusts image so its mean luminosity matches target_lum.
    
    Parameters:
        image_path (str): Path to the input image.
        target_lum (float): Target average luminosity (0–255).
        save_path (str, optional): If given, saves adjusted image here.
    
    Returns:
        PIL.Image: Adjusted image.
    """
    # Open image and convert to RGB
    arr = np.array(img, dtype=np.float32)

    # Convert to grayscale (luminosity = weighted sum of RGB)
    # Rec. 709 luma coefficients
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    current_lum = np.mean(lum)

    # Compute scaling factor
    scale = target_lum / current_lum if current_lum > 0 else 1.0

    # Apply scaling to RGB channels
    adjusted = arr * scale
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(adjusted, mode="RGB")

    return out_img

def gatherData(dataset):
    typesOfData = {
        "Park_Healthy" : "Parkinson_Drawings/drawings/spiral/training/healthy/",
        "Park_Park": "Parkinson_Drawings/drawings/spiral/training/parkinson/",
        "Hand_Park" : "HandPD/Spiral_HandPD/SpiralPatients/",
        "Hand_Healthy" :  "HandPD/Spiral_HandPD/SpiralControl/",
        "New_Healthy" : "NewHandPD/HealthySpiral/",
        "New_Park" :"NewHandPD/PatientSpiral/",
        }
    d = "/projects/NLS_ADPIE/data/"
    if (dataset == "Parkinson_Drawings"):
        return loadImages(typesOfData["Park_Healthy"], "CTL", d) | loadImages(typesOfData["Park_Park"], "PD", d)
    elif (dataset == "HandPD"):
        return loadImages(typesOfData["Hand_Healthy"], "CTL", d) | loadImages(typesOfData["Hand_Park"],"PD", d)
    
    elif (dataset == "NewHandPD"):
        return loadImages(typesOfData["New_Healthy"], "CTL", d) | loadImages(typesOfData["New_Park"], "PD", d)
    
    elif (dataset == "PaHaW"):
        return getPaHaW(d)
    
    elif (dataset == "NLS"):
        return getNLS(d)
    
    elif (dataset == "all_img"):
        a = loadImages(typesOfData["Park_Healthy"], "CTL_park", d, crossCopra=True) | loadImages(typesOfData["Park_Park"], "PD_park", d, crossCopra=True)
        b =  loadImages(typesOfData["Hand_Healthy"], "CTL_nhp", d, crossCopra=True) | loadImages(typesOfData["Hand_Park"],"PD_nhp", d, crossCopra=True)
        c = loadImages(typesOfData["New_Healthy"], "CTL_hp", d, crossCopra=True) | loadImages(typesOfData["New_Park"], "PD_hp", d, crossCopra=True)
        
        return a | b | c
    
    elif (dataset == "all_ts"):
        return gatherDataNLS(dataset)
    elif (dataset == "all"):

        return gatherData("all_img") | gatherData("all_ts")
    elif (dataset == "spirals"):
        return gatherDataNLS("spirals")
    
    elif (dataset == "nls_all"):
        return gatherDataNLS("all")

    elif (dataset == "all_spirals"):
        return gatherDataNLS("spirals") | gatherData("all_img") | gatherData("PaHaW")
    
    else:
        print(f"Invalid dataset name. {dataset} Only valid datasets are: ")
        print("\tParkinson_Drawings")
        print("\tHandPD")
        print("\tNewHandPD")
        print("\tPaHaW")
        print("\tNLS")
        return None
    
def gatherDataNLS(task):
    if (task == "points"):
        return getNLS(r"(point_DOM|point_NONDOM|point_sustained)")
    elif (task == "spirals"):
        return getNLS(r"(spiral_DOM|spiral_NONDOM|spiral_pataka)")
    elif (task == "numbers"):
        return getNLS(r"(numbers)")
    elif (task == "writing"):
        return getNLS(r"(copytext|copyreadtext|freewrite)")
    elif (task == "drawing"):
        return getNLS(r"(drawclock|copycube|copymage)")
    elif (task == "all"):
        keywords = r"point_DOM|point_NONDOM|point_sustained|spiral_DOM|spiral_NONDOM|spiral_pataka|numbers|"
        keywords += r"copytext|copyreadtext|freewrite|drawclock|copycube|copymage"
        return getNLS(keywords)
    elif (task == "all_ts"):
        return gatherDataNLS("all") | getPaHaW("/projects/NLS_ADPIE/data/")
    else:
        return getNLS(task)

def xy2img(df, hasBS=False, isPoint=False):
    """
    Function to convert x,y coordinates to an image
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with x,y coordinates
    hasBS : bool
        Boolean to determine if the data has a button press
    Returns
    -------
    img : PIL image
        Image with x,y coordinates
    """
    # Extract X, Y, and Pressure values
    X = df['X'].astype(float).values
    Y = df['Y'].astype(float).values
    P = df['P'].astype(float).values

    # Normalize Pressure values to [0, 1]
    P_min, P_max = np.min(P), np.max(P)
    P_s = ((P - P_min) / (P_max - P_min)) ** 0.5 if (P_max > P_min) else np.ones_like(P)
    #P_a = ((P - P_min) / (P_max - P_min)) if (P_max > P_min) else np.ones_like(P)

    # If hasBS is True, filter points where BS == 1
    if hasBS and 'BS' in df.columns:
        S = df['BS'].astype(bool).values
        X, Y, P_s = X[S], Y[S], P_s[S]
        del S

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)  # High-resolution image
    ax.axis("off")  # Hide axis
    
    if isPoint:
        mask = P != 0
        C = ['black' if m else 'blue' for m in mask]
        ax.set_xlim(12000,18000)
        ax.set_ylim(6000,10000)
        ax.scatter(X, Y, c=C, alpha=0.5, s=1)
        del mask, C
    else:
        ax.set_xlim(np.min(X) - 5, np.max(X) + 5)  # Add some padding
        ax.set_ylim(np.min(Y) - 5, np.max(Y) + 5)
         # Scatter plot with transparency based on P
        # ax.scatter(X, Y, c='black', alpha=P, s=1)
        ax.scatter(X, Y, c='black', alpha=0.1, s=P_s)


    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    
    # Free Matplotlib figure memory
    plt.close(fig)  
    del fig, ax  # Delete figure objects explicitly

    # Convert buffer to PIL image
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    
    # Free buffer memory
    buf.close()
    del buf  

    # Free NumPy arrays
    del X, Y, P_s  

    return img

def getPaHaW(data_name= "/../../projects/NLS_ADPIE/data/"):
    """
    Function to load PaHaW data
    Parameters
    ----------
    data_name : str
        Path to the data
    Returns
    -------
    pahaw_data : dict
        Dictionary with keys as image names and values as images
    """
    # First Gather Disease info for each patient
    file_path = "/projects/NLS_ADPIE/data/PaHaW/PaHaW_files/corpus_PaHaW.xlsx"
    df = pd.read_excel(file_path)
    df['Disease'] = df['Disease'].replace("H", "CTL")
    
    ID_TYPE = pd.Series(df['Disease'].values, index=df['ID']).to_dict()

    # Second Iterate through folders to get svc files
    directory = "/projects/NLS_ADPIE/data/PaHaW/PaHaW_public/"
    pahaw_data = {}
    column_names =['Y', 'X', 'T', 'BS', 'Az', 'Al', 'P']
    countP = 1
    countC = 1
    for folder in sorted(os.listdir(directory)): # Iterate through folders
        folder_path = os.path.join(directory, folder)
        name = ID_TYPE[int(folder)] + '_' + str(int(folder))
        spiralType = findLabel(name)
        for svc_file in sorted(os.listdir(folder_path)):
            if spiralType=="PD":
                svc_paht = os.path.join(folder_path, svc_file)
                df = pd.read_csv(svc_paht, sep=' ', skiprows=1, header=None, names=column_names)
                img = xy2img(df,hasBS=True)
                pahaw_data[f'{spiralType}__{countP}'] = {"image": img, "label": spiralType}
                countP+=1
            elif spiralType=="CTL":
                svc_paht = os.path.join(folder_path, svc_file)
                df = pd.read_csv(svc_paht, sep=' ', skiprows=1, header=None, names=column_names)
                img = xy2img(df,hasBS=True)
                pahaw_data[f'{spiralType}__{countC}'] = {"image": img, "label": spiralType}
                countC+=1
    return pahaw_data

def getNLS(_labels):
    """
    Function to load NLS data
    Parameters
    ----------
    data_name : str
        Path to the data
    Returns
    -------
    nls_data : dict
        Dictionary with keys as image names and values as images
    """
    # First Gather Disease info for each patient
    file_path = "/projects/NLS_ADPIE/data/NLS/handwriting/clean/0.metadata.csv"
    df = pd.read_csv(file_path)
    # Goal is to only extract PD and control

    df = df[(df['label'] == Config.LABEL_1) | (df['label'] == Config.LABEL_2)]
    # Create Dictionary
    ID_TYPE = pd.Series(df['label'].values, index=df['ID']).to_dict()

    # Second Iterate through folders to get svc files
    directory = "/projects/NLS_ADPIE/data/NLS/handwriting/clean/"
    nls_data = {}
    valid_labels = Config.NLS_DATA
    countP = 1
    countC = 1
    isPoint = "point" in _labels
    for folder in sorted(os.listdir(directory)): # Iterate through folders
        if folder in ID_TYPE.keys() and (ID_TYPE[folder] in valid_labels):
            folder_path = os.path.join(directory, folder)
            taskType = ID_TYPE[folder] # Determines the disease type (AD, PD, PDM, CTL, etc.) of each fie
            for svc_file in sorted(os.listdir(folder_path)): 
                if re.search(_labels, svc_file) and taskType==Config.LABEL_1: # True if label is PD
                    svc_paht = os.path.join(folder_path, svc_file)
                    df = pd.read_csv(svc_paht)
                    img = xy2img(df, isPoint=isPoint)
                    nls_data[f'{taskType}_{countP}_{svc_file}'] = {"image": img, "label": Config.LABEL_1}
                    countP += 1 
                elif re.search(_labels, svc_file) and taskType==Config.LABEL_2: # True if label is CTL
                    svc_paht = os.path.join(folder_path, svc_file)
                    df = pd.read_csv(svc_paht)
                    img = xy2img(df, isPoint=isPoint)
                    nls_data[f'{taskType}_{countC}_{svc_file}'] = {"image": img, "label": Config.LABEL_2}
                    countC += 1 
    return nls_data

def findLabel(filename):
    if ("PD" in filename):
        return "PD"
    elif("CTL" in filename):
        return "CTL"
    else:
        print("Error in Finding Label: ", filename)
        os.exit()
        
def partitionData(data):
    """
    Converts the data into a dictionary with 'X' and 'y' keys. 

    Parameters
    ----------
    data: dictionary
        Dictionary of the data that holds an image and label for each participant.

    Returns
    ---------
    Dictionary of X and y keys where X is the images and y is a binary value 0 or 1. 

    """
    X = [d["image"] for d in data.values()]
    y = [d["label"] for d in data.values()]
    map_labels = lambda labels: np.array([{Config.LABEL_1: 0, Config.LABEL_2: 1}[label] for label in labels])
    
    return {"X": X, "y": map_labels(y)}

######################### MODEL AND TRAINING FUNCTIONS #########################

class ResNetWithMLP(nn.Module):
    def __init__(self, base_model_name="microsoft/resnet-50", mlp_dims=[512, 128, 2]):
        super().__init__()
        self.base_model = ResNetForImageClassification.from_pretrained(base_model_name)

        # Determine base output dim
        classifier = self.base_model.classifier
        if isinstance(classifier, nn.Sequential):
            for layer in reversed(classifier):
                if isinstance(layer, nn.Linear):
                    base_output_dim = layer.in_features  # input to classifier, not out_features
                    break
        elif isinstance(classifier, nn.Linear):
            base_output_dim = classifier.in_features
        else:
            raise ValueError(f"Unsupported classifier type: {type(classifier)}")

        # Replace classifier with Identity
        self.base_model.classifier = nn.Identity()

        # Build MLP
        layers = []
        in_dim = base_output_dim
        for dim in mlp_dims[:-1]:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        layers.append(nn.Linear(in_dim, mlp_dims[-1]))
        
        # Final Output lawyer
        #layers.append(nn.Softmax(dim=1))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        features = outputs.logits if hasattr(outputs, "logits") else outputs
        features = features.view(features.size(0), -1)
        logits = self.mlp_head(features)
        return logits


def train_one_fold(train_loader, val_loader, device, num_epochs, model, patience=10, lr=1e-4):
    """
    Train a model for one fold of cross-validation with early stopping.

    This function trains the given model using a training dataloader and evaluates 
    it on a validation dataloader each epoch. It supports HuggingFace models (with 
    `.logits`) or custom models that return raw outputs. Training uses AdamW as the 
    optimizer and cross-entropy loss. Early stopping is applied based on validation 
    loss to prevent overfitting, and the best-performing model weights are restored 
    before returning.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation set.
    device : torch.device
        Device to run the training on (e.g., 'cuda' or 'cpu').
    num_epochs : int
        Maximum number of training epochs.
    model : torch.nn.Module
        The neural network model to train.
    patience : int, optional (default=10)
        Number of epochs to wait for validation loss improvement before early stopping.
    lr : float, optional (default=1e-4)
        Learning rate for the AdamW optimizer.

    Returns
    -------
    model : torch.nn.Module
        The trained model with weights restored to the best validation performance.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state, counter = None, 0

    for epoch in range(num_epochs):
        # ── Training loop ──
        model.train()
        running_loss = 0.0
        for px, lbl in train_loader:
            px, lbl = px.to(device), lbl.to(device)
            out = model(px)

            # Handle HuggingFace vs custom model outputs
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validation loop ──
        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for px, lbl in val_loader:
                px, lbl = px.to(device), lbl.to(device)
                out = model(px)
                logits = out.logits if hasattr(out, "logits") else out

                loss = criterion(logits, lbl)
                val_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)

        val_acc = correct / total
        print(f"E{epoch+1:02d}  train-loss {running_loss:.4f}  val-loss {val_loss:.4f}  val-acc {val_acc:.4f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop on epoch {epoch+1}")
                break
    model.load_state_dict(best_state)  # Reload best state
    
    return model 

def preprocess_images(processor, image_list, labels):
    """
    Preprocess a batch of images and corresponding labels into tensors.

    Each image in `image_list` is processed using the given `processor` 
    (e.g., a HuggingFace image processor) to generate normalized pixel 
    values suitable for input into a model. The images are stacked into 
    a single tensor, and the labels are converted into a tensor of dtype `long`.

    Parameters
    ----------
    processor : callable
        A preprocessing function or HuggingFace image processor that takes 
        an image and returns a dictionary containing "pixel_values".
    image_list : numpy.ndarray
        Array of images to preprocess.
    labels : numpy.ndarray
        Array of integer labels, where 1 corresponds to Config.Label1 
        and 0 corresponds to Config.Label2.

    Returns
    -------
    images_tensor : torch.Tensor
        Tensor of shape (N, C, H, W) containing the processed images.
    labels_tensor : torch.Tensor
        Tensor of shape (N,) containing the labels as class indices.
    """
    tensors = []
    for img in image_list:
        inputs = processor(img, return_tensors="pt")["pixel_values"]
        tensors.append(inputs.squeeze(0))  # Remove batch dim
    images_tensor = torch.stack(tensors) 

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor

######################### TESTING FUNCTIONS #########################

def testCNN(model, X_test, y_test):
    """
    Test the trained CNN on test data.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN/ResNet model
    processor : transformers.AutoImageProcessor
        Preprocessing pipeline
    test_data : dict
        Test image data {"X": [PIL images], "y": labels}

    Returns
    -------
    accuracy : float
        Classification accuracy on test set
    auc : float
        AUC score (only for binary classification)
    f1 : float
        F1 score (weighted average)
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=X_test.to(device))
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = torch.argmax(logits, dim=1)

    # Move to CPU for metric calculations
    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_prob = torch.softmax(logits, dim=1).detach().cpu().numpy()

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # F1 Score (weighted for multi-class)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # AUC Score
    try:
        if y_prob.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:  # Multiclass
            print("multiclass error")
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        auc = float('nan')  # e.g. if only one class is present in y_true

    print(f"Test Accuracy: {accuracy:.3%}")
    print(f"Test F1 Score: {f1:.3f}")
    print(f"Test AUC Score: {auc:.3f}")

    return accuracy, auc, f1



######################### MAIN FUNCTIONS #########################

def train_and_get_model(d_trainval, task, model_name=Config.MODEL_NAME, 
                        batch_size=Config.BATCH_SIZE, num_epochs=Config.NUM_EPOCHS, 
                        n_splits=Config.N_SPLITS, patience=Config.PATIENCE, 
                        random_state=Config.RANDOM_STATE, testing_data=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)


    X_full, y_full = np.array(d_trainval["X"]), np.array(d_trainval["y"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#     all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []},
#                    "HandPD" : {"accuracy": [], "auc": [], "f1_score": []},
#                    "NewHandPD": {"accuracy": [], "auc": [], "f1_score": []},
#                    "Parkinson_Drawings": {"accuracy": [], "auc": [], "f1_score": []},
#                    "PaHaW": {"accuracy": [], "auc": [], "f1_score": []}
#                    }
    #["all", "spirals", "spiral_DOM", "spiral_NONDOM", "spiral_pataka"]
    #all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []},
    #               "all_pahaw" : {"accuracy": [], "auc": [], "f1_score": []}
#                    ,"spirals": {"accuracy": [], "auc": [], "f1_score": []},
#                    "spiral_DOM": {"accuracy": [], "auc": [], "f1_score": []},
#                    "spiral_NONDOM": {"accuracy": [], "auc": [], "f1_score": []},
#                    "spiral_pataka": {"accuracy": [], "auc": [], "f1_score": []}
#                    }
#    data_sets = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW", "spirals", "nls_all"]
    # all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []},
    #                "HandPD" : {"accuracy": [], "auc": [], "f1_score": []},
    #                "NewHandPD": {"accuracy": [], "auc": [], "f1_score": []},
    #                "Parkinson_Drawings": {"accuracy": [], "auc": [], "f1_score": []},
    #                "PaHaW": {"accuracy": [], "auc": [], "f1_score": []},
    #                "spirals": {"accuracy": [], "auc": [], "f1_score": []},
    #                "nls_all": {"accuracy": [], "auc": [], "f1_score": []}
    # 
    # 
    #                }
    if Config.RUN_COPRA:
        task = f"MODEL_WITHOUT_{task}"
        all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []},
                    "HandPD" : {"accuracy": [], "auc": [], "f1_score": []},
                    "NewHandPD": {"accuracy": [], "auc": [], "f1_score": []},
                    "Parkinson_Drawings": {"accuracy": [], "auc": [], "f1_score": []},
                    "PaHaW": {"accuracy": [], "auc": [], "f1_score": []},
                    "spirals": {"accuracy": [], "auc": [], "f1_score": []},
                    "nls_all": {"accuracy": [], "auc": [], "f1_score": []}
                }
    elif Config.TEST_TRANS:
        all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []},
                    "HandPD" : {"accuracy": [], "auc": [], "f1_score": []},
                    "NewHandPD": {"accuracy": [], "auc": [], "f1_score": []},
                    "Parkinson_Drawings": {"accuracy": [], "auc": [], "f1_score": []},
                    "PaHaW": {"accuracy": [], "auc": [], "f1_score": []},
                    "spirals": {"accuracy": [], "auc": [], "f1_score": []},
                    "nls_all": {"accuracy": [], "auc": [], "f1_score": []}
                }
    else:
        all_metrics = {task: {"accuracy": [], "auc": [], "f1_score": []}}
        

    if Config.SHOW_DATA_SPLITS:
        count_PD = lambda labels: np.count_nonzero(labels)
        count_CTL = lambda labels: labels.shape[0] - count_PD(labels)

    if Config.TEST_TRANS or Config.RUN_COPRA:
        new_test_datasets = load_additional_tests(testing_data)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n────────── Fold {fold + 1}/{n_splits} ──────────")
        
        # Split data
        X_full_raw, y_full_raw = X_full[train_idx], y_full[train_idx]
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_full_raw, y_full_raw, test_size=0.2, random_state=random_state, stratify=y_full_raw)
        X_test_raw, y_test_raw = X_full[test_idx], y_full[test_idx]
        
        if Config.SHOW_DATA_SPLITS:
            print(f"\n TRAIN: PD-{count_PD(y_train_raw)} CTL-{count_CTL(y_train_raw)}")
            print(f"\n VAL: PD-{count_PD(y_val_raw)} CTL-{count_CTL(y_val_raw)}")
            print(f"\n TEST: PD-{count_PD(y_full[test_idx])} CTL-{count_CTL(y_full[test_idx])}")
            print()


        X_train, y_train = preprocess_images(processor, X_train_raw, y_train_raw)
        X_val, y_val = preprocess_images(processor, X_val_raw, y_val_raw)
        X_test, y_test = preprocess_images(processor, X_test_raw, y_test_raw)

        train_loader = DataLoader(TensorDataset(X_train, y_train.long()), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val.long()), batch_size=batch_size)

        model = ResNetWithMLP().to(device)
        # Train one fold
        model = train_one_fold(train_loader, val_loader, device, num_epochs, model=model, patience=patience)
        
        # Evaluate on test set
        accuracy, auc, f1 = testCNN(model, X_test, y_test)
        all_metrics[task]["accuracy"].append(accuracy)
        all_metrics[task]["auc"].append(auc)
        all_metrics[task]["f1_score"].append(f1)

        if Config.TEST_TRANS or Config.RUN_COPRA:
        # external datasets (store results too if you want them later)
            for test_name, test in new_test_datasets.items():
                #if (test_name != task):
                name = "fold" + str(fold+1)
                test_fold = test[name]
                X, y = test_fold["X"], test_fold['y']
                X, y = preprocess_images(processor, X, y)
                acc, auc, f1 = testCNN(model, X, y)
                all_metrics[test_name]["accuracy"].append(acc)
                all_metrics[test_name]["auc"].append(auc) 
                all_metrics[test_name]["f1_score"].append(f1)

        del train_loader, val_loader, model
        torch.cuda.empty_cache()
    
    return all_metrics

def process_data_and_train(task, processing_function=None, data=None, testing_data=None):
    """
    Processes data, trains the ResNet and SVM models, and saves them.
    
    Args:
        task (str): The name of the task to process.
        processing_function: The function to gather the data for the task.
    """
    print(f"\nProcessing task: {task}")
    if processing_function is not None:
        data = processing_function(task)
    dtrainval = partitionData(data) 
    all_metrics = train_and_get_model(dtrainval, task, testing_data=testing_data)
    
    if Config.TEST_TRANS:
        for cur_task, metrics in all_metrics.items():
            avg_accuracy = np.mean(metrics["accuracy"])
            avg_auc = np.mean(metrics["auc"])
            avg_f1 = np.mean(metrics["f1_score"])
            std_acc = np.std(metrics["accuracy"])
            
            #print each metric above
            print(f"Task: {cur_task} - Average Accuracy + std: {avg_accuracy:.4f} +- {std_acc}, Average AUC: {avg_auc:.4f}, Average F1 Score: {avg_f1:.4f}")

        return all_metrics

    else: 
        metrics = all_metrics[task]
        avg_accuracy = np.mean(metrics["accuracy"])
        avg_auc = np.mean(metrics["auc"])
        avg_f1 = np.mean(metrics["f1_score"])
        std_acc = np.std(metrics["accuracy"])

        #print each metric above
        print(f"Task: {task} - Average Accuracy + std: {avg_accuracy:.4f} +- {std_acc}, Average AUC: {avg_auc:.4f}, Average F1 Score: {avg_f1:.4f}")

        return {"accuracy": avg_accuracy, "auc": avg_auc, "f1_score": avg_f1}

def load_additional_tests(data_name=None, data=None) -> dict:
    new_test_datasets = {}
    if data is None:
        data_sets = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW", "spirals", "nls_all"]
        for set_name in data_sets:
                data = gatherData(set_name)
                test = partitionData(data)
                skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=42)
                test_fold = {}
                num_folds = Config.N_SPLITS
                for fold, (_,test_idx) in enumerate(skf.split(test["X"], test["y"])):
                    X_test, y_test = np.array(test["X"])[test_idx], np.array(test["y"])[test_idx]
                    test_fold["fold" + str(fold+1)] = {"X": X_test, "y": y_test}
                
                new_test_datasets[set_name] = test_fold

    else:
        test = partitionData(data)
        skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=42)
        test_fold = {}
        num_folds = Config.N_SPLITS
        for fold, (_,test_idx) in enumerate(skf.split(test["X"], test["y"])):
            X_test, y_test = np.array(test["X"])[test_idx], np.array(test["y"])[test_idx]
            test_fold["fold" + str(fold+1)] = {"X": X_test, "y": y_test}
        
        new_test_datasets[data_name] = test_fold

    return  new_test_datasets 

def display_metrics_table(all_metrics: dict):
    """
    Display averaged metrics in a neatly formatted table (no external libraries).

    Parameters
    ----------
    all_metrics : dict
        Expected format:
        {task: {"accuracy": float, "auc": float, "f1_score": float}}
    """
    headers = ["Task", "Accuracy", "AUC", "F1 Score"]

    # Prepare rows
    rows = []
    for task, metrics in all_metrics.items():
        rows.append([
            task,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['auc']:.4f}",
            f"{metrics['f1_score']:.4f}"
        ])

    # Compute column widths (max length across header + rows)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build row formatter
    row_fmt = " | ".join("{:<" + str(w) + "}" for w in col_widths)

    # Print header
    print(row_fmt.format(*headers))
    print("-+-".join("-" * w for w in col_widths))

    # Print rows
    for row in rows:
        print(row_fmt.format(*row))


######## MAIN EXECUTION ########
if __name__ == "__main__":
    #multiprocessing.set_start_method('spawn', force=True)

    # Ensure this is the correct path for the user's system
    # It's better to use an absolute path or a robust relative path
    try:
        os.chdir(Config.BASE_PATH)
        print(f"Current working directory: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Warning: The base path {Config.BASE_PATH} was not found. Please set the correct path.")
        exit()
    
    if Config.RUN_COPRA:
        name = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW", "spirals"]

        dataHPD = gatherData("HandPD")
        dataNHPD = gatherData("NewHandPD")
        dataPD = gatherData("Parkinson_Drawings")
        dataPW = gatherData("PaHaW")
        dataNLS = gatherDataNLS("spirals")

        data = {
            "HandPD" : dataHPD,
            "NewHandPD": dataNHPD,
            "Parkinson_Drawings": dataPD,
            "PaHaW": dataPW,
            "spirals" : dataNLS
            }

        for i in range(5):
            print(f"TESTING ON {name[i]}")
            DO_NOT_USE = {name[i]}
            new_Dataset = list(DO_NOT_USE ^ set(name))
            final_data = {}
            for key, value in data.items():
                final_data = final_data | value

            process_data_and_train(task=name[i], data=final_data, testing_data=data[name[i]])

    else:
        all_tasks = ["points", "spirals", "numbers", "writing", "drawing", "all"]
        individual_tasks = [
            "point_DOM", "point_NONDOM", "point_sustained",
            "spiral_DOM", "spiral_NONDOM", "spiral_pataka",
            "numbers", "copytext", "copyreadtext", "freewrite",
            "drawclock", "copycube", "copymage"
        ]
        data_sets = ["HandPD", "NewHandPD", "Parkinson_Drawings", "PaHaW"]

        large_metric = {}
        # Process each task
        print("-" * 60)
        print(f"{"ALL TASKS":^60}")
        print("-" * 60)
        for task in all_tasks:
            print(f"{task:*^60}")
            metrics = process_data_and_train(task, gatherDataNLS)
            large_metric[task] = metrics
        
        # Process individual tasks
        print("-" * 60)
        print(f"{"INDIVIDUAL TASKS":^60}")
        print("-" * 60)
        for task in individual_tasks:
            print(f"{task:*^60}")
            metrics = process_data_and_train(task, gatherDataNLS)
            large_metric[task] = metrics
        
        # process datasets
        print("-" * 60)
        print(f"{"DATASETS":^60}")
        print("-" * 60)
        for dataset in data_sets:
            print(f"{dataset:*^60}")
            metrics = process_data_and_train(dataset, gatherData)
            large_metric[dataset] = metrics
 
        display_metrics_table(large_metric)
