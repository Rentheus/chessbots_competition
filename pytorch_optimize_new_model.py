import torch
from torch import nn
import numpy as np
import chess
from sklearn.model_selection import train_test_split
from pathlib import Path


pieces = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
colours = [chess.WHITE,chess.BLACK]

def fenToVec(fen):
	posFen = fen.split()[0]
	board = chess.BaseBoard(posFen)
	l = []
	
	for colour in colours:
		for piece in pieces:
			v = np.zeros(64)
			for i in list(board.pieces(piece,colour)):
				v[i] = 1
			v = v.reshape((8,8))
			l.append(v)
	#
	return np.array(l)

#print(fenToVec("1rbbq3/p3k3/3ppP2/1p6/PQ4P1/1N6/1PP1P1PR/RN2KB2 b Q - 0 19").shape)

arr = np.loadtxt("posval.csv2", delimiter=",", dtype=str)

positions = []
value = []

for i in arr:
	positions.append(torch.tensor(fenToVec(i[0])))
	value.append(torch.tensor(float(i[1])))
	
positions = torch.from_numpy(np.array(positions)).type(torch.float)
value = torch.unsqueeze(torch.tensor(value).type(torch.float),1)

#print(positions.shape)
#print(value.shape)


pos_train, pos_test, val_train, val_test = train_test_split(positions, value, test_size=0.2,)

print(pos_train.size(), val_train.size(),pos_test.size(),val_test.size())

#m = nn.Conv2d(12, 22, 4 )
#print(m(positions[0]).size())
#n = nn.Flatten()
#print(n(m(positions[0])).size())
#f = nn.Flatten()
#print(f(n(m(positions[0]))).size())

model_2 = nn.Sequential(
	nn.Conv2d(12, 22, 4 ),
	nn.Flatten(), #22x5x5
	nn.LeakyReLU(),
	nn.Linear(550, 100),
	nn.Mish(),
	nn.Linear(100,64),
	nn.LeakyReLU(),
	nn.Linear(64,1)
	
)

model_3 = nn.Sequential(
	nn.Conv2d(12, 22, 4 ),
	nn.Flatten(), #22x5x5
	nn.Tanh(),
	nn.Linear(550, 100),
	nn.Tanh(),
	nn.Linear(100,64),
	nn.LeakyReLU(),
	nn.Linear(64,4),
	nn.Linear(4,1)
	
)

model_4 = nn.Sequential(
	nn.Conv2d(12, 22, 4 ),
	nn.Flatten(), #22x5x5
	nn.LeakyReLU(),
	nn.Linear(550, 2400),
	nn.LeakyReLU(),
	nn.Linear(2400,64),
	nn.LeakyReLU(),
	nn.Linear(64,4),
	nn.Linear(4,1)
	
)



loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_4.parameters(), lr = 0.015)


epochs= 20000
train_loss_values = []
test_loss_values = []
epoch_count = []


for epoch in range(epochs):
    model_4.train()
    val_pred = model_4(pos_train)
    loss = loss_fn(val_pred, val_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_4.eval()
	


    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_4(pos_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, val_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
			
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_4.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_4.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 