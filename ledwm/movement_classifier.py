# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set device to CUDA if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Sentence Transformer model
# model_name = "sentence-transformers/all-mpnet-base-v2"
# t5
# model = SentenceTransformer("sentence-transformers/sentence-t5-base")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model_name = "sentence-transformers/all-MiniLM-L12-v2"
embedder = SentenceTransformer(model_name).to(device)

# Load the training dataset
train_dataset_path = (
    "train_movement.csv"  # Replace with the path to your training dataset CSV
)
df_train = pd.read_csv(train_dataset_path)

# Load the validation dataset
val_dataset_path = (
    "val_movement.csv"  # Replace with the path to your validation dataset CSV
)
df_val = pd.read_csv(val_dataset_path)

# Filter out rows where the 'Category' is 'unknown'
df_train = df_train[df_train["Category"].str.lower() != "unknown"].reset_index(
    drop=True
)
df_val = df_val[df_val["Category"].str.lower() != "unknown"].reset_index(drop=True)


# Encode text data using SentenceTransformer
def encode_texts(texts):
    embeddings = embedder.encode(texts, convert_to_tensor=True, device=device)
    return embeddings


# Encode the labels using LabelEncoder based on training data
label_encoder = LabelEncoder()


df_train["EncodedCategory"] = label_encoder.fit_transform(df_train["Category"])
df_val["EncodedCategory"] = label_encoder.transform(
    df_val["Category"]
)  # Ensure consistency with training labels

# Encode train and test texts
train_embeddings = encode_texts(df_train["Text"].tolist())
val_embeddings = encode_texts(df_val["Text"].tolist())


# Create PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# Create DataLoader for training and validation
train_dataset = TextDataset(
    train_embeddings, torch.tensor(df_train["EncodedCategory"].tolist()).to(device)
)
val_dataset = TextDataset(
    val_embeddings, torch.tensor(df_val["EncodedCategory"].tolist()).to(device)
)

# %%
# val_dataset[0]
# %%

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Define a simple Linear Classification model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# Initialize the model, loss function, and optimizer
input_dim = train_embeddings.shape[1]
output_dim = len(label_encoder.classes_)
model = LinearClassifier(input_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}"
        )


# Evaluate the model
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in val_loader:
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


# Run training and evaluation
train_model(model, train_loader, criterion, optimizer, num_epochs=100)
evaluate_model(model, val_loader)
