import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



transform_train = transforms.Compose( [ transforms.RandomResizedCrop(360, scale=(0.9, 1.0)), transforms.RandomHorizontalFlip(p=0.5), 
                                       transforms.RandomRotation(15), 
                                        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ] )  

transform_test = transforms.Compose( [ transforms.Resize((360, 360)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
transform_valid = transforms.Compose( [ transforms.Resize((360, 360)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
#the normalize tool maps image values from [0,1] to [-1, 1] which is standard for deep learning models 
#Compose combines multiple image transformstions sequentially 
#ToTensor converts PIL image values (0-255) to pytorch tensors(0-1)
#Normalize normalizes each rgb channel where mean=0.5 and std=0.5 (pixel = (pixel-mean)/std). now it maps the pixel values from [0,1] to [-1,1]



train_data = datasets.ImageFolder(root=r"D:\Honey\Fun\cancer detection using sam 2\code\Data\train", transform=transform_train) 
test_data = datasets.ImageFolder(root=r"D:\Honey\Fun\cancer detection using sam 2\code\Data\test", transform=transform_test) 
val_data = datasets.ImageFolder(root=r"D:\Honey\Fun\cancer detection using sam 2\code\Data\valid", transform=transform_valid) 
#loading the training, test and validation data using imagefolder



train_loader = DataLoader(train_data, batch_size=16, shuffle=True) 
test_loader = DataLoader(test_data, batch_size=16, shuffle=False) 
val_loader = DataLoader(val_data, batch_size=16, shuffle=False) 
#creating data loaders to load images in batches of 16

print(train_data.classes) 
num_classes = len(train_data.classes)



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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



model = models.densenet121(weights = "IMAGENET1K_V1")

#for param in model.features.parameters():                                              #freezing entire feature extractor
#    param.requires_grad = False

#for param in model.features.denseblock4. parameters():                                 #unfreezing last denseblock
#    param.requires_grad = True

for param in model.features[-12:].parameters():                                          #unfreezing last dense block and transitional layer
    param.requires_grad = True


num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, num_classes))                                 #replacing the classifier

model = model.to(device)



class_weights = torch.tensor([2.625, 6.176, 5.833, 3.500], dtype=torch.float).to(device)                 #balancing class weights

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)



num_epochs = 20
best_loss = float("inf")

for epoch in range(num_epochs):                                                     #training loop
    model.train()
    train_loss = 0.0
    train_total = 0.0
    train_correct = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds==labels).sum().item()

    train_acc = 100*train_correct / train_total
    train_loss /= len(train_loader)



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
    scheduler.step(val_loss)



    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")


    print(f"Epoch: [{epoch+1}/{num_epochs}]     training loss: {train_loss:.4f}     training accuracy: {train_acc:.2f}%     validation loss: {val_loss:.4f}     validation accuracy: {val_acc:.2f}%")



model.eval()
all_preds = []
all_labels = []
all_logits = []

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    
    _, preds = torch.max(outputs, 1) 
    probs = torch.softmax(outputs, dim=1)
    
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy()) 
    all_logits.extend(probs.cpu().numpy()) 

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test accuracy: {accuracy*100:.2f}%")



from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Classification report
report = classification_report(all_labels, all_preds, target_names=train_data.classes)
print(report)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_data.classes, yticklabels=train_data.classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show() 



from sklearn.metrics import roc_auc_score 
from sklearn.preprocessing import label_binarize 

label_true = label_binarize(all_labels, classes=[0, 1, 2, 3]) 
label_prob = np.array(all_logits)

roc_auc = roc_auc_score(label_true, label_prob, average="macro") 
print("roc_auc score: ", roc_auc)



from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget 
from PIL import Image

def gradcam_explain(model, image_path): 
    model.eval() 

    img = Image.open(image_path).convert("RGB") 
    input_tensor = transform_test(img).unsqueeze(0).to(device) 

    target_layers = [model.features.denseblock4] 

    cam = GradCAM(model=model, target_layers=target_layers) 
    targets = [ClassifierOutputTarget(torch.argmax(model(input_tensor)).item())] 

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0] 

    rgb_image = np.array(img.resize((360,360))) / 255.0 
    viz = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True) 

    plt.figure(figsize=(6,6)) 
    plt.imshow(viz) 
    plt.axis("off") 
    plt.title("model attention using grad cam") 
    plt.show()  

image_path = r"D:\Honey\Fun\cancer detection using sam 2\code\Data\test\large.cell.carcinoma\000110.png" 
gradcam_explain(model, image_path)



def predict_single_image(model, image_path, device=device):
    """
    Returns:
        {
            'prediction': 'class_name',
            'probabilities': {
                              class1: prob,
                              class2: prob,
                              class3: prob,
                              class4: prob
                            }
        }
    """

    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        probs = probs.cpu().numpy()[0]

        prob_dict = {class_names[i]: float(probs[i])
                       for i in range(len(class_names))}

        top_class = class_names[np.argmax(probs)]

    return {
        "prediction": top_class,
        "probabilities": prob_dict
      }


def load_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(weights="IMAGENET1K_V1")
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model 


