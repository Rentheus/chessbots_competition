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


CHESS_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}
M_DEPTH = 3
M_INF = 999

def evaluate_board(board: chess.Board, color: chess.Color) -> int:
    score = 0

    for piece_type, value in CHESS_VALUES.items():
        score += (
            len(board.pieces(piece_type, color)) - \
                len(board.pieces(piece_type, not color))
        ) * value

    return score

def mm(board: chess.Board, depth, alpha, beta, max_p, color):
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return (M_INF if color == chess.WHITE else -M_INF) * depth
        if result == "0-1":
            return (M_INF if color == chess.BLACK else -M_INF) * depth
        return 0
    if depth == 1:
        return evaluate_board(board, color)

    if max_p:
        m_eval = -M_INF
        for move in board.legal_moves:
            board.push(move)
            c_eval = mm(board, depth - 1, alpha, beta, False, color)
            board.pop()
            m_eval = max(m_eval, c_eval)
            alpha = max(alpha, c_eval)
            if beta <= alpha:
                break
        return m_eval
    m_eval = M_INF
    for move in board.legal_moves:
        board.push(move)
        c_eval = mm(board, depth - 1, alpha, beta, True, color)
        board.pop()
        m_eval = min(m_eval, c_eval)
        beta = min(beta, c_eval)
        if beta <= alpha:
            break
    return m_eval

def fick_sie_alle_bot(shared, board: chess.Board) -> None:
    best_score = -M_INF
    color = board.turn

    for move in board.legal_moves:
        board.push(move)
        score = mm(board, M_DEPTH, -M_INF, M_INF, False, color) + random.random()/2
        if board.is_check():
            score += 0.5
        board.pop()
        # print(move, score)
        mmove = chess.Move.from_uci(str(move))
        if board.piece_at(mmove.from_square).symbol() == \
            chess.Piece(chess.KING, board.turn).symbol():
            score -= 0.5
            if chess.square_distance(mmove.from_square, mmove.to_square) > 1:
                score += 1.5

        if score > best_score:
            best_score = score
            shared.best_move = move



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


            score = -loaded_model_0(torch.tensor(vec).type(torch.float))

            if score > best_score:
                best_score = score
                shared.best_move = move
                print(score)
            board.pop()
