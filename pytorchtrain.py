#
import torch
from torch import nn
import numpy as np
import chess
from sklearn.model_selection import train_test_split
from pathlib import Path



pieces = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
colours = [chess.WHITE,chess.BLACK]

class board_eval_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.linear(in_features = 768, out_features = 32)
        self.layer_2 = nn.linear(in_features = 32, out_features = 4)
        self.layer_3 = nn.linear(in_features = 4, out_features = 1)
        
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_3(self.layer_2(self.layer_1(x))) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

	
	


def fen_to_matr(fen):
    board = chess.Board(fen)
    pieces = board.piece_map()
    return pieces

def fenToVec(fen):
	posFen = fen.split()[0]
	board = chess.BaseBoard(posFen)
	l = []
	
	for colour in colours:
		for piece in pieces:
			v = np.zeros(64)
			for i in list(board.pieces(piece,colour)):
				v[i] = 1
			l.append(v)
	l = np.concatenate(l)
	return l

arr = np.loadtxt("posval.csv", delimiter=",", dtype=str)

positions = []
value = []

for i in arr:
	positions.append(torch.tensor(fenToVec(i[0])))
	value.append(torch.tensor(float(i[1])))
	
positions = torch.from_numpy(np.array(positions)).type(torch.float)
value = torch.unsqueeze(torch.tensor(value).type(torch.float),1)
#print(positions[:5])


pos_train, pos_test, val_train, val_test = train_test_split(positions, 
                                                    value, 
                                                    test_size=0.15,)

pos_train1, pos_train2, val_train1, val_train2,  = train_test_split(pos_train, 
                                                    val_train, 
                                                    test_size=0.5,)

print(pos_train1.size(), val_train1.size(),pos_test.size(),val_test.size())


model_0 = nn.Sequential(
    #nn.Conv2d(768,768, 3),
    #nn.Conv2d(12, 32, kernel_size=3, padding=1),
    nn.Linear(768,133),
    nn.LogSigmoid(),
    nn.Linear(133,137),
    nn.SELU(),
    nn.Linear(137,16),
    nn.LeakyReLU(),
    nn.Linear(16,1)
   # nn.Linear(in_features=768, out_features=300),
   # nn.LeakyReLU(),
   # nn.Linear(in_features=300, out_features=300),
   # nn.Linear(in_features=300, out_features=200),
   # nn.LeakyReLU(),
   # nn.Linear(in_features=200, out_features=300),
   # nn.Linear(in_features=300, out_features=300),
   # nn.Linear(in_features=300, out_features=128),
   # nn.LeakyReLU(),
   # nn.Linear(in_features=128, out_features=32),
   # nn.LeakyReLU(),
   # nn.Linear(in_features=32, out_features=32),
   # nn.Linear(in_features=32, out_features=1)
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.005)


epochs= 1900
train_loss_values = []
test_loss_values = []
epoch_count = []


for epoch in range(epochs):
    model_0.train()
    val_pred1 = model_0(pos_train1)
    loss = loss_fn(val_pred1, val_train1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()

    val_pred2 = model_0(pos_train2)
    loss = loss_fn(val_pred2, val_train2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()


    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(pos_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, val_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
      #if test_loss < 6: 
       #     break
#print(fenToVec("8/8/1k2K2R/4PP2/8/8/7p/7r w - - 0 59"))

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 