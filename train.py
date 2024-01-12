import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time
import torch
from torch import optim as optim
import torchvision
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Train a model on a dataset")


from torch import nn as nn
def get_arg():
    parser.add_argument("data_directory", type=str, help="Path to the data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth", help="Directory to save the checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (vgg16 or vgg13)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    
    
def data_aug(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Done: Define your transforms for the training, validation, and testing sets
    transformers = {'train' : transforms.Compose([transforms.RandomRotation(10),
                                           transforms.Resize(size=[224,224]),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]
                                         ),
                       'test' : transforms.Compose([transforms.Resize(size=[224,224]),
                                                    transforms.CenterCrop(224),   
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]
                                         )}
    # Done: Load the datasets with ImageFolder
    dataset = { 'train' : datasets.ImageFolder(train_dir,transform=transformers['train']),
                'test'  : datasets.ImageFolder(test_dir,transform=transformers['test']),
                'valid' : datasets.ImageFolder(valid_dir,transform=transformers['test'])}

    # Done: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train' : torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True),
                    'test' : torch.utils.data.DataLoader(dataset['test'], batch_size=64),
                    'valid': torch.utils.data.DataLoader(dataset['valid'], batch_size=64)}
    return dataloaders , dataset

def choose_model(version):
    if version == 'vgg13':
        model = models.vgg13(pretrained=True)
    else :
        model = models.vgg16(pretrained=True)
    return model

def create_classifier(hidden_units):
    classifier =  nn.Sequential(nn.Linear(25088, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_units, 102),
                            nn.LogSoftmax(dim=1))
    return classifier

def model_training_validation(model , training_dataset , validation_dataset , criterion,optimizer , epochs , device , learn_rate ,valid_every = 5 ):

    steps = 0

    running_loss = 0
    
    model.to(device);

    for epoch in range(epochs):
        for inputs, labels in training_dataset:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % valid_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validation_dataset:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/valid_every:.3f}.. "
                      f"Test loss: {test_loss/len(validation_dataset):.3f}.. "
                      f"Test accuracy: {(accuracy/len(validation_dataset))*100:.3f}")
                running_loss = 0
                model.train()
    print("TRAINING IS FINISHED")
    
    
def model_testing(model , testing_dataset , criterion , device):
    
    test_loss = 0
    accuracy = 0
    
    #Evaluating the model !! 
    model.eval()
    model.to(device);
    with torch.no_grad():
        for inputs, labels in testing_dataset:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        print(f"The Final test loss: {test_loss/len(testing_dataset):.3f}.. "
                      f"Test accuracy: {(accuracy/len(testing_dataset))*100:.3f}")
        
        
        
        
def saving_checkpoint(saving_path , training_dataset , model , epochs , optimizer , learn_rate ):
    args = parser.parse_args()
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier' : model.classifier,
                  'class_to_idx' : training_dataset.class_to_idx,
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'epochs' : epochs,
                  'learning_rate' : learn_rate,
                  'model_version' :args.arch 
                 }
    torch.save(checkpoint, saving_path)
    print("Checkpoint is saved successfully !!")
    
    
    
def build_model():
    
    args = parser.parse_args()
    
    data_dir = args.data_directory
    
    dataloaders , datasets = data_aug(data_dir)
    
    model = choose_model(args.arch)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = create_classifier(args.hidden_units)

    # Set model Classifier 
    model.classifier = classifier
    
    device = "cuda" if args.gpu else "cpu"
    
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    valid_every = 5
    
    model_training_validation(model , dataloaders['train'],dataloaders['valid'],criterion ,optimizer,args.epochs,device,args.learning_rate)
    
    model_testing(model , dataloaders['test'],criterion,device)
    
    saving_checkpoint(args.save_dir,datasets['train'],model,args.epochs,optimizer,args.learning_rate)

def main():
    get_arg()
    build_model()
    print("Model is build successfully !")

    
if __name__ == "__main__":
    main()

