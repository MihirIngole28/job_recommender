# src/ranking_model.py
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

class RankingNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_ranking_model(features, labels, epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = RankingNN(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
        targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # Eval
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    ndcg = ndcg_score([y_test], [test_outputs.squeeze().cpu().numpy()], k=10)
    print(f"NDCG@10: {ndcg}")
    torch.save(model.state_dict(), 'models/ranking_model.pt')
    return model

# Example: Features = [cosine_sim, skill_overlap, location_match]; labels = relevance (synthetic 0-1)
if __name__ == "__main__":
    # Generate dummy features/labels from data
    features = np.random.rand(1000, 3)  # Replace with real (e.g., cosine + one-hot)
    labels = np.random.rand(1000)
    model = train_ranking_model(features, labels)