'''
PyTorch to build a neural network using a dataset from UC Irvine ML repository.
Below code is from 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic' which shows thew dataset
split into two parts for easy workings.
Data set is full and informative.
'''
from ucimlrepo import fetch_ucirepo
from IPython.core.display_functions import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


'''
#1. first have to sort sort the data, separate the answers (b or m) from the data
#2. make sure the data is even and balanced, i.e. need 100 of each not 150/200 etc
#3. standardise the data so its close to mean and not far apart. X-mean / std which puts all values inbetween -1 and 1. convert to tensors
#4. build the model
#5. train the model
'''


# downloaded the data set due to slow loading from the UC repo
X = pd.read_csv("features.csv")
y = pd.read_csv("targets.csv")

#display(X.head()) # Shows we have  30 different features
#display(y.head()) # Shows diagnosis. Either M or B

#display(X.shape, y.shape) # shows us 569 samples in the data set and 30 features and 1 label for y
#display(y['Diagnosis'].value_counts()) # gives B=357 and M = 212. Unbalanced so cap at 200 each

data = pd.concat([X, y], axis=1) # combine to 1 pd dataframe, stack beside to align

# need to separate the two class into b and m and select 20 random samples from each and combine
data_B = data[data['Diagnosis'] == 'B']
data_M = data[data['Diagnosis'] == 'M']

data_B = data_B.sample(n=200, random_state=959) # take the 200 random and store the set
data_M = data_M.sample(n=200, random_state=959)
balanced_data = pd.concat([data_B, data_M])
#display(balanced_data['Diagnosis'].value_counts())  # 200 for both B and M, successfully balanced the data

#### preprocess data ready for the model
X = balanced_data.drop('Diagnosis', axis=1) # get rid of the diagnosis column in X (features set)
y = balanced_data['Diagnosis'].map({'B': 0, 'M': 1}) # store the diagnosis into y which is our target and map/convert to 0/1 for B,M so it'll work with the model
# display(y) # shows 0's and 1's

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=959, stratify=y) # stratify to maintain the ratio in both test and train sets. Keeps balance
# display(X_train.shape) # shows data split 80/20 (320/80) for the train and test sets
# display(y_train.shape)
# display(X_test.shape)
# display(y_test.shape)

scaler = StandardScaler() # initialise the scaler for use. 0 mean, 1 var
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # transform the train and test data (features)

X_train = torch.tensor(X_train, dtype = torch.float32) # convert to tensors, float for precision and long for the labels.
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train.values, dtype = torch.long)
y_test = torch.tensor(y_test.values, dtype = torch.long)

train_dataset = TensorDataset(X_train, y_train) # pairs tensor features with label , treated as obj
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # load data in batches instead of all at once.
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class classificationNetwork(nn.Module):
    def __init__(self, input_units = 30, hidden_units = 64, output_units = 2):
        super(classificationNetwork, self).__init__() # use to call the nn constructor
        self.fc1 = nn.Linear(input_units, hidden_units) # fully connected layer , transforms the input to hidden
        self.fc2 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # pass x through layer 1 and uses relu activation, for more complex patterns
        x = self.fc2(x) # pass through 2nd layer
        return x # raw logits

model = classificationNetwork(input_units = 30, hidden_units = 64, output_units = 2) # network instance

criterion = nn.CrossEntropyLoss() # this uses soft max to convert the raw logits
optimizer = optim.Adam(model.parameters(), lr=0.001) # use adam to update weights during training


#training
epochs = 10
train_losses, test_losses = [], [] #store loss values
for epoch in range(epochs):
    model.train() # training phase/ enables gradients
    running_loss = 0.0 # track for each epoch
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad() # clearing past gradients
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward() # derivative
        optimizer.step() # update weights
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss) # avg for each epoch

    model.eval() # eval mode, doesn't update gradients
    test_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():  # No gradient computation
        for X_batch, y_batch in test_loader:
            test_outputs = model(X_batch)  # Get logits
            loss = criterion(test_outputs, y_batch)  # Compute test loss
            test_loss += loss.item()

            # Convert logits to predicted class
            logit, predicted = torch.max(test_outputs, 1) # max along columns
            predictions.extend(predicted.numpy())  # Store predictions
            labels.extend(y_batch.numpy())  # Store actual labels

    # Compute average test loss
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    # Compute accuracy
    accuracy = sum(1 for x, y in zip(predictions, labels) if x == y) / len(labels)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')


plt.figure(figsize=(12, 8))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss curve')
plt.legend()
plt.show()

# Generate confusion matrix
conf_matrix = confusion_matrix(labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues',
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()
