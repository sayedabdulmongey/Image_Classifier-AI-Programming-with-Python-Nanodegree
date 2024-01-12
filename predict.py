import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time
import torch
from torch import nn as nn
from torch import optim as optim
import torchvision
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import argparse
import json


parser = argparse.ArgumentParser(description="Predict using a trained model")

def get_arg():
    
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to the mapping of categories to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for prediction")

    
def read_categories(category_path):
    with open(category_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint['model_version']=='vgg13':
        model = models.vgg13(pretrained=True)
    else :
        model = models.vgg16(pretrained=True)
        
   
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model , checkpoint

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    target_size=256
    
    # determine the shortest side
    shortest_side = min(img.width, img.height)

    # calculate the new size with the specified target size
    if img.width < img.height:
        new_width = target_size
        new_height = int(target_size * (img.height / img.width))
    else:
        new_width = int(target_size * (img.width / img.height))
        new_height = target_size

    # resize the image
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    # cropping out the center 224 x 224 
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = (img.width + 224) / 2
    bottom = (img.height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    image_array = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = image_array/[255.0,255.0,255.0]
    normalized_image=(image_array-mean) / std
    
    final_image = normalized_image.transpose((2,0,1))
    
    return final_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image_path, model, device , topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Done: Implement the code to predict the class from an image file

    model.to(device);
    
    processed_image=torch.tensor(process_image(image_path),dtype=torch.float32)
    processed_image.unsqueeze_(0)
    processed_image = processed_image.to(device)
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(processed_image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p , top_class
def predict_print_image(image_path,model,cat_to_name,device,top_k):
    
    prob , classes= predict(image_path,model,device,top_k)

    labels=[cat_to_name[str(j)] for i in classes.cpu().numpy() for j in i]

    prob=prob.cpu().numpy().flatten()
    
    for i in range(top_k):
        print(f"Class {labels[i]} : {prob[i]*100:0.3f} %")
    print(f"The most likely image class is : {labels[np.argmax(prob)]} and its probability is : {prob[np.argmax(prob)]*100:.3f} %")
def model_prediction():
    
    args = parser.parse_args()
    
    cat_to_name = read_categories(args.category_names) 
    
    image_path = args.input 
    
    device = "cuda" if args.gpu else "cpu"
    
    model , checkpoint = load_checkpoint(args.checkpoint)
    
    predict_print_image(image_path , model , cat_to_name,device ,args.top_k )
    
    
def main():
    get_arg()
    model_prediction()
        
    
if __name__ == "__main__":
    main()
    
