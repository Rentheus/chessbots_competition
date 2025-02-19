"""
create your own bot here
"""

import random
import math
import chess
from functools import lru_cache

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

lru_cache(maxsize=250)
def legal_moves(board: chess.Board) -> list:
    return list(board.legal_moves)



def evaluate_board(board: chess.Board, color: chess.Color) -> int:
    score = 0

    for piece_type, value in CHESS_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
        score -= len(board.pieces(piece_type, not color)) * value

    return score

def mm(board: chess.Board, depth, alpha, beta, max_p, color):
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return math.inf if color == chess.WHITE else -math.inf
        if result == "0-1":
            return math.inf if color == chess.BLACK else -math.inf
        return 0
    if not depth:
        return evaluate_board(board, color)

    if max_p:
        m_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            c_eval = mm(board, depth - 1, alpha, beta, False, color)
            board.pop()
            m_eval = max(m_eval, c_eval)
            alpha = max(alpha, c_eval)
            if beta <= alpha:
                break
        return m_eval
    m_eval = math.inf
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
    best_score = -math.inf
    color = board.turn

    for move in board.legal_moves:
        board.push(move)
        score = mm(board, 2, -math.inf, math.inf, False, color) + random.random()/2
        if board.is_check():
            score += 0.5
        board.pop()

        mmove = chess.Move.from_uci(str(move))
        if board.piece_at(mmove.from_square).symbol() == \
            chess.Piece(chess.KING, board.turn).symbol():
            score -= 0.5
            if chess.square_distance(mmove.from_square, mmove.to_square) > 1:
                score += 1.5

        if score > best_score:
            best_score = score
            shared.best_move = move
