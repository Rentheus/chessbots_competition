"""
create your own bot here
"""

import random
import math
import chess


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
