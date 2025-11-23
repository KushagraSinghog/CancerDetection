from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

transform = transforms.Compose( [ transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
#the normalize tool maps image values from [0,1] to [-1, 1] which is standard for deep learning models 
#Compose combines multiple image transformstions sequentially 
#ToTensor converts PIL image values (0-255) to pytorch tensors(0-1)
#Normalize normalizes each rgb channel where mean=0.5 and std=0.5 (pixel = (pixel-mean)/std). now it maps the pixel values from [0,1] to [-1,1]



train_data = datasets.ImageFolder(root=r'E:\Honey\Fun\cancer detection using sam 2\code\Data\train', transform=transform) 
test_data = datasets.ImageFolder(root=r'E:\Honey\Fun\cancer detection using sam 2\code\Data\test', transform=transform) 
val_data = datasets.ImageFolder(root=r'E:\Honey\Fun\cancer detection using sam 2\code\Data\valid', transform=transform) 
#loading the training, test and validation data using imagefolder 



train_loader = DataLoader(train_data, batch_size=50, shuffle=True) 
test_loader = DataLoader(test_data, batch_size=50, shuffle=False) 
val_loader = DataLoader(val_data, batch_size=50, shuffle=False) 
#creating data loaders to load images in batches of 50

print(train_data.classes)  



import matplotlib.pyplot as plt 
import numpy as np 
import torchvision 

def imshow(img):                                                        #function to unnormalize and display images
    img = img/2 + 0.5                                                   #unnormalizes the images and converts the tensors from [-1,1] to [0,1]
    npimg = img.numpy()                                                 #converts tensors to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))                          #transpose swaps axises from (C,H,W) to (H,W,C)
    plt.show()  



dataiter = iter(train_loader)                                           #creates an iterator for training images
images, labels = next(dataiter)                                         #gets a batch of images and labels

imshow(torchvision.utils.make_grid(images[:8]))                             #combines multiple images into a single grid (8) and displays them
print('Labels: ', [train_data.classes[i] for i in labels[:8]])                    #prints image labels of the 8 images 



import torch
import torch.nn as nn
import torch.nn.functional as F

class cancer_pred_cnn(nn.Module):
    def __init__(self, num_classes=4):
        super(cancer_pred_cnn, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = cancer_pred_cnn(num_classes=4).to(device)  

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 



num_epochs = 20
for epoch in range(num_epochs):                                                           #training loop
    model.train() 
    running_loss = 0.0 
    total = 0.0                                                                           #total no of images going through the model
    correct = 0.0                                                                         #no of images correctly predicted
    
    for images, labels in train_loader:                                                   #loop to load the data batchwise 
        images, labels = images.to(device), labels.to(device) 
        optimizer.zero_grad()                                                             
        outputs = model(images)                                                           
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item()                                                       #converts loss from tensor to float 
        total += labels.size(0)                                                           #adds no of images in batch to total
        _, predicted = torch.max(outputs.data, 1)                                         #returns index of class with highest scores instead of a tuple
        correct += (predicted == labels).sum().item()                                     #counts no of correct predictions

    train_loss = running_loss / len(train_loader)                                         #avg epoch loss
    train_acc = 100*correct/total                                                         
   
   
   
    model.eval() 
    val_correct=0.0 
    val_total=0.0 
    val_loss=0.0 

    with torch.no_grad(): 
        for images, labels in val_loader: 
            images, labels = images.to(device), labels.to(device) 
            outputs=model(images) 
            loss = criterion(outputs, labels) 

            val_loss += loss.item() 
            _, predicted = outputs.max(1) 
            val_total += labels.size(0) 
            val_correct += predicted.eq(labels).sum().item() 

    val_acc = 100*val_correct/val_total 
    val_loss /= len(val_loader)

    print(f"Epoch: [{epoch+1}/{num_epochs}]     training loss: {train_loss:.4f}     training accuracy: {train_acc:.2f}%     validation loss: {val_loss:.4f}     validation accuracy: {val_acc:.2f}%")
        

