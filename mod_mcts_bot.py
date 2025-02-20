import random
import math
import chess
import numpy as np
import time

M_DEPTH = 3
M_INF = 999

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




val_field_pawn = np.flip(np.array([
    1,1,1,1,1,1,1,1,
    5,5,5,5,5,5,5,5,
    2,2,2,2,2,2,2,2,
    1,2,1,3,3,1,2,1,
    2,1,2,3,3,2,1,2,
    1,2,1,2,2,1,2,1,
    1,1,2,1,1,2,1,1,
    1,1,1,1,1,1,1,1,
])) * 1/10 + 1

val_field_knight = np.flip(np.array([
    1,1,1,1,1,1,1,1,
    1,2,2,2,2,2,2,1,
    1,2,3,3,3,3,2,1,
    1,2,3,2,2,3,2,1,
    1,2,3,2,2,3,2,1,
    1,2,3,3,3,3,2,1,
    1,2,2,2,2,2,2,1,
    1,1,1,1,1,1,1,1,
])) * 1/10 + 3

val_field_bishop = np.flip(np.array([
    1,1,1,1,1,1,1,1,
    1,3,1,1,1,1,3,1,
    1,3,1,1,1,1,3,1,
    1,3,3,1,1,3,3,1,
    1,3,3,1,1,3,3,1,
    1,3,1,1,1,1,3,1,
    1,3,1,1,1,1,3,1,
    1,1,1,1,1,1,1,1,
])) * 1/10 + 3.2

val_field_rook = np.flip(np.array([
    3,3,3,3,3,3,3,3,
    1,1,1,1,1,1,1,1,
    1,1,1,2,2,1,1,1,
    1,2,1,1,1,1,2,1,
    1,2,1,1,1,1,2,1,
    1,1,3,1,1,3,1,1,
    1,1,3,2,2,3,1,1,
    1,1,3,3,3,3,1,1,
])) * 1/10 + 5

val_field_queen = np.flip(np.array([
    1,1,2,2,2,2,1,1,
    1,3,2,4,4,2,3,1,
    1,3,2,4,4,2,3,1,
    1,3,2,4,4,2,3,1,
    1,3,2,4,4,2,3,1,
    1,3,2,4,4,2,3,1,
    1,1,2,2,2,2,2,1,
    1,1,1,1,1,1,1,1,
])) * 1/10 + 9.5

val_field_king = np.flip(np.array([
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,
    0,1,0,0,0,0,1,0,
    1,0,0,0,0,0,0,1,
    1,0,0,0,0,0,0,1,
    0,1,1,0,0,1,1,0,
    1,2,2,1,1,2,2,1,
])) * 1/10 

pieces = [chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN,chess.KING]

val_board = [val_field_pawn,val_field_knight,val_field_bishop,val_field_rook,val_field_queen,val_field_king]

def evaluate_board(eboard: chess.Board):
    'always evaluates as white, for eval as black pass board.copy().mirror()'
    #score : nmoves & pos.dependent scores & pinning
    nmoves = eboard.legal_moves.count()
    nscore = 2*np.log(nmoves + 0.01)
    
    bonscore = 0
    #bonus score: rooks, queen on on same rank/file (todo), both knights/bishops alive, king not on file/rank of enemy rook, check if queen attacket by enemy
    bonscore += (3//(len(list(eboard.pieces(chess.KNIGHT, chess.WHITE)))+1))*1/4
    bonscore += (3//(len(list(eboard.pieces(chess.BISHOP, chess.WHITE)))+1))*1/4
    if len(list(eboard.pieces(chess.QUEEN, chess.WHITE)))>0:
        bonscore -= 5*eboard.is_attacked_by(chess.BLACK, list(eboard.pieces(chess.QUEEN, chess.WHITE))[0])
    if len(list(eboard.pieces(chess.QUEEN, chess.BLACK)))>0:
        bonscore -= 5*eboard.is_attacked_by(chess.WHITE, list(eboard.pieces(chess.QUEEN, chess.BLACK))[0])
    


    bonscore -= 1/3*(eboard.pieces(chess.ROOK, chess.BLACK).issubset(chess.SquareSet(chess.BB_RANKS[eboard.king(chess.WHITE)%8]|chess.BB_FILES[eboard.king(chess.WHITE)//8])))

    #white score:
    wscore = 0
    #print(list(eboard.pieces(5, chess.WHITE)).append(1))
    for p in pieces:
        #print(p)
        wscore += sum(val_board[p-1][np.array(list(eboard.pieces(p, chess.WHITE))+[1])]) - val_board[p-1][1]
    #black_score:
    eboard = eboard.mirror()

    bscore = 0
    for p in pieces:
        bscore += sum(val_board[p-1][np.array(list(eboard.pieces(p, chess.WHITE))+[1])]) - val_board[p-1][1]
    #print(nscore, bonscore, wscore, bscore)
    return nscore + bonscore + wscore - bscore

results = {
    "1-0": +20,
    "1/2-1/2": -1,
    "0-1": -20,
}


def mod_mcts_bot(shared,  board: chess.Board):
    """
    example of a better bot that makes moves based on the number of pieces
    """
    if board.turn == chess.BLACK:
        board = board.mirror()
    legal_moves = np.array(list(board.legal_moves))
    scores = np.zeros(legal_moves.shape)
    #print(len(legal_moves))
    n_explorations = int(1/(0.021* len(scores)/20))
    print(n_explorations)
    for j in range(len(scores)):
        tboard = board.copy()
        tboard.push(legal_moves[j])
    
        scores[j] -= n_explorations*1/3* evaluate_board(tboard.copy().mirror())
        for n_expl in range(n_explorations):
            ttboard = tboard.copy()
            for i in range(7):
                legal_list = list(ttboard.legal_moves)
                if len(legal_list) == 0:
                    scores[j] += results[ttboard.outcome().result()]
                    break
                ttboard.push(random.choice(legal_list))
                if i%2 == 0:
                    scores[j] += evaluate_board(ttboard)
                else:
                    scores[j] -= evaluate_board(ttboard.copy().mirror())
        scores[j] += random.random()/2
        shared.best_move = legal_moves[scores == max(scores)][0]
    #print(legal_moves)
    #return best_move
    


#b  = chess.Board("r1bq2nr/pppppk1p/5bp1/8/P3P3/5N2/1PP2PPP/RNB1K2R w KQ - 2 8")
#t1 = time.time()
#print(mod_mcts_bot(b))
#t2 = time.time()
#print(t2-t1)


def mm(board: chess.Board, depth, alpha, beta, max_p, color):
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return (M_INF if color == chess.WHITE else -M_INF) * depth
        if result == "0-1":
            return (M_INF if color == chess.BLACK else -M_INF) * depth
        return 0
    if depth == 1:
        if color == False:
            return -1* evaluate_board(board.copy().mirror())
        else:
            return evaluate_board(board)

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

def geklauter_bot_bessere_eval(shared, board: chess.Board) -> None:
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
