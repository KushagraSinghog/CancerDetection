                                      Lung Cancer Detection using DenseNet-121

A deep learning project that classifies lung CT images into four categories:

1. Adenocarcinoma
2. Large Cell Carcinoma
3. Squamous Cell Carcinoma
4. Normal Lung Tissue

This model is trained using DenseNet-121 with transfer learning, achieving ~91% accuracy on the test dataset. The project includes data preprocessing, training, evaluation, visualization. 


Key Features:

1. Four-Class Lung Cancer Classification: The model predicts the probability (%) of each class and provides the final predicted label.

2. DenseNet-121 (Transfer Learning): Uses ImageNet-pretrained DenseNet-121 with a custom classification head.

3. Class Balancing: Weighted cross-entropy loss to improve training on imbalanced cancer types.

4. Strong Performance:
                       Test Accuracy: ~91%
                       Macro F1-Score: ~0.91

   Stable loss and accuracy curves with minimal overfitting.

6. Production-Ready Deployment:

    Includes a Gradio interface for:

      Uploading JPG images
      Displaying predicted class
      Showing probability distribution


| Component     | Technology                         |
| ------------- | -----------------------------------|
| DL Framework  | PyTorch, Torchvision               |
| Model         | DenseNet-121 (ImageNet pretrained) |
| Deployment    | Gradio, Hugging Face Spaces        |
| Visualization | Matplotlib, Seaborn                |
| Data Handling | PyTorch DataLoader, ImageFolder    | 




Training Pipeline: 

   1. Data Augmentation: RandomResizedCrop, Horizontal Flip, Rotation, Resize + Normalize

   2. Model Modification: Unfreezing last DenseNet block, Adding custom classifier 

   3. Optimizer & Scheduler: Adam (LR = 1e-4), ReduceLROnPlateau scheduler

   4. Loss function: criterion = nn.CrossEntropyLoss(weight=class_weights)



Results: 

   | Class                | Precision | Recall | F1-Score |
   | -------------------- | --------- | ------ | -------- |
   | Adenocarcinoma       | 0.90      | 0.86   | 0.88     |
   | Large Cell Carcinoma | 0.81      | 0.86   | 0.84     |
   | Normal               | 1.00      | 0.98   | 0.99     |
   | Squamous Cell        | 0.93      | 0.97   | 0.95     |

   Overall Accuracy: 91.11%
   Macro F1-Score: 0.91 
