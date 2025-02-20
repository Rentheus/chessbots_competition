
import random
import math
import chess
import numpy as np
#import time




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
