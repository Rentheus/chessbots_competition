"""
create your own bot here
"""

import random
import math
import chess
import torch
from torch import nn
import numpy as np



def bad_bot(shared, board: chess.Board):
    """
    example of a bad bot that makes random moves
    """
    shared.best_move = random.choice(list(board.legal_moves))


def my_bot(shared, board: chess.Board):
    """
    example of a better bot that makes moves based on the number of pieces
    """
    best_score = -math.inf
    for move in board.legal_moves:
        board.push(move)
        score = 0
        for piece in board.piece_map().values(): # does not take into account the actual piece-values
            if piece.color != board.turn:
                score += piece.piece_type
            else:
                score -= piece.piece_type
        score += random.random()
        if score > best_score:
            best_score = score
            shared.best_move = move
        board.pop()

def fenToVec(fen):
    pieces = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
    colours = [chess.WHITE,chess.BLACK]
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

def better_rand_bot(shared, board: chess.Board):
    """
    example of a better bot that makes moves based on the number of pieces
    """


    best_score = -math.inf
    loaded_model_0 = nn.Sequential(
    nn.Linear(768,133),
    nn.LogSigmoid(),
    nn.Linear(133,137),
    nn.SELU(),
    nn.Linear(137,16),
    nn.LeakyReLU(),
    nn.Linear(16,1)
                )
    loaded_model_0.load_state_dict(torch.load(f="models/01_pytorch_workflow_model_0.pth"))
    loaded_model_0.eval()
    with torch.inference_mode():

        for move in board.legal_moves:
            board.push(move)
            score = 0
            vec = fenToVec(board.fen())

        
            score = -(loaded_model_0(torch.tensor(vec).type(torch.float))).item()

            if score > best_score:
                best_score = score
                shared.best_move = move
                print(score)
            board.pop()


def better_rand_bot2(shared, board: chess.Board):
    """
    example of a better bot that makes moves based on the number of pieces
    """


    best_score = -math.inf
    loaded_model_0 = nn.Sequential(
    nn.Linear(768,133),
    nn.Mish(),
    nn.Linear(133,64),
    nn.Mish(),
    nn.Linear(64,16),
    nn.Mish(),
    nn.Linear(16,1)
                )
    loaded_model_0.load_state_dict(torch.load(f="models/01_pytorch_workflow_model_1.pth"))
    loaded_model_0.eval()
    with torch.inference_mode():

        for move in board.legal_moves:
            board.push(move)
            score = 0
            vec = fenToVec(board.fen())

        
            score = -(loaded_model_0(torch.tensor(vec).type(torch.float))).item()

            if score > best_score:
                best_score = score
                shared.best_move = move
                #print(score)
            board.pop()

def fenToVec2(fen):
     
    pieces = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]
    colours = [chess.WHITE,chess.BLACK]
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

def better_rand_bot3(shared, board: chess.Board):
    """
    example of a better bot that makes moves based on the number of pieces
    """


    best_score = -math.inf
    loaded_model_0 = nn.Sequential(
	nn.Conv2d(12, 22, 4 ),
	nn.Flatten(0,-1), #22x5x5
	nn.LeakyReLU(),
	nn.Linear(550, 100),
	nn.Mish(),
	nn.Linear(100,64),
	nn.LeakyReLU(),
	nn.Linear(64,1)
                )
    loaded_model_0.load_state_dict(torch.load(f="models/01_pytorch_workflow_model_2.pth"))
    loaded_model_0.eval()
    with torch.inference_mode():

        for move in board.legal_moves:
            board.push(move)
            score = 0
            vec = torch.tensor(fenToVec2(board.fen())).type(torch.float)
            #print(vec.size())
        
            score = -(loaded_model_0(vec)).item()

            if score > best_score:
                best_score = score
                shared.best_move = move
                #print(score)
            board.pop()