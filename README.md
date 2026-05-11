# Developing a Neural Network Classification Model
### Date: 27-04-2026
## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="828" height="830" alt="image" src="https://github.com/user-attachments/assets/05f5086b-8689-47cb-af02-be363231bfa6" />




## DESIGN STEPS
### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

### STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

### STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

### STEP 6:
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: THARUN D

### Register Number: 212223240167

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self, x):
      #Include your code here
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        
# Initialize model
model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1068" height="223" alt="Screenshot 2026-05-11 132339" src="https://github.com/user-attachments/assets/53bc3880-fd74-4845-8e4a-490795bcc5e7" />


### OUTPUT

## Confusion Matrix

<img width="711" height="462" alt="Screenshot 2026-05-11 132421" src="https://github.com/user-attachments/assets/a281f516-1e3e-485e-b81f-3c648ebea7c4" />


## Classification Report
<img width="569" height="358" alt="Screenshot 2026-05-11 132413" src="https://github.com/user-attachments/assets/b688c999-fbeb-49cd-8042-47fe7e37137e" />


### New Sample Data Prediction
<img width="387" height="96" alt="Screenshot 2026-05-11 132527" src="https://github.com/user-attachments/assets/b8017a76-cc0d-4c42-a8c9-4d06124b0570" />

## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
