import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

DATA_PATH="data_processed"

class ChessValueDataset(Dataset):
    def __init__(self):
        data = np.load(f'{DATA_PATH}/train_set_100k.npz')
        self.X = data['arr_0']
        self.y = data['arr_1']
        print(f"Loaded data: X: {self.X.shape}, Y: {self.y.shape} ({self.X.shape[0]} samples).")

    def __len__(self):
        # of samples
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def get_data(self):
        return self.X, self.y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            # 8x8 chess board
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            # 4x4
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            # 2x2
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2), 
            nn.ReLU(),

            # 1x128
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),      

            # flatten
            nn.Flatten(),
            nn.Linear(128, 1),
            # Activation
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

EPOCHS=20
BATCH_SIZE=256
MAX_BATCHES=None
print_interval = 5

chess_dataset = ChessValueDataset()
trainloader = torch.utils.data.DataLoader(chess_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Net()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
if (torch.cuda.is_available()):
    print("Training on GPU.")
else:
    print("Training on CPU")

def train(trainloader, EPOCHS, MAX_BATCHES=None):
	print(f"Starting training with {len(trainloader)} batches") 
	
	writer = SummaryWriter("logs")
	num_steps_test = 20
	running_loss = 0

	losses = []

	for epoch in range(EPOCHS):
		for batch_idx, (X, y) in enumerate(tqdm(trainloader)): 
			if MAX_BATCHES is not None and batch_idx > MAX_BATCHES:
				return

			y = y.unsqueeze(-1)
			X, y = X.to(device), y.to(device) 
			X = X.float()
			y = y.float()
			optimizer.zero_grad()
			output = model(X)
			loss = loss_function(output, y)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			losses.append(loss)

			if batch_idx % print_interval == 0:
				print(f"[e{epoch}/{EPOCHS}] b({batch_idx*len(X)/len(trainloader.dataset)}%) - loss: {loss.item():.4f} ")

			# test on validation data
			if batch_idx % num_steps_test == num_steps_test-1:
				writer.add_scalar('Loss/training', running_loss/num_steps_test, (batch_idx+1)*(epoch+1))
				running_loss = 0

	torch.save(model.state_dict(), f"{root_path}/MODEL-e{EPOCHS}-s{len(trainloader.dataset)}.pth")

	# plot loss data at the end of training
	plt.figure(figsize=(10,5))
	plt.title("chess.ai Training Loss")
	plt.plot(losses, label="Training loss")
	plt.xlabel("steps")
	plt.ylabel("loss")
	plt.legend()
	plt.show()

if __name__ == "__main__":
    train(trainloader, EPOCHS)
