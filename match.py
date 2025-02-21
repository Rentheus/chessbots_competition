"""
main file to test the custom bots
"""

import threading
import ctypes
import random
import chess
import chess.pgn
from datetime import datetime
from custom_bot import bad_bot, my_bot, better_rand_bot, fick_sie_alle_bot


BOTS = {
    "bad_bot"           : bad_bot,
    "my_bot"            : my_bot,
    "better_rand_bot"   : better_rand_bot,
    "fick_sie_alle_bot" : fick_sie_alle_bot,
}

class ChessBot:
    def __init__(self, logic_func, name = "Bot") -> None:
        self.logic_func = logic_func
        self.name = name
        self.shared = type('Namespace', (), {'best_move': None})()
        self.error_count = [0, 0, 0]

    def generate_move(self, board: chess.Board, time_limit: int = 5) -> chess.Move:
        self.shared.best_move = None

        def worker():
            try:
                self.logic_func(self.shared, board.copy())
            except SystemExit:
                pass
            except Exception as e:
                print(f"Error in bot execution: {e}")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=time_limit)

        if thread.is_alive():
            print("Time limit exceeded. Using best move found so far...")
            self.error_count[0] += 1
            exc = ctypes.py_object(SystemExit)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), exc)
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)

        if self.shared.best_move is None:
            print("No valid move found. Choosing random move...")
            self.error_count[1] += 1
            return random.choice(list(board.legal_moves))
        if self.shared.best_move not in board.legal_moves:
            print(f"Invalid move ({self.shared.best_move}). Choosing random move...")
            self.error_count[2] += 1
            return random.choice(list(board.legal_moves))

        return self.shared.best_move


def print_board(board):
    print(board)
    print(board.fen())
    print("\n\n")


def main():
    board = chess.Board()

    game = chess.pgn.Game()
    game.headers["Event"] = "CHESSBOT_COMPETITON"
    game.headers["White"] = "fick_sie_alle_bot"
    game.headers["Black"] = "better_rand_bot"
    game.headers["Date"] = datetime.now()
    node = game

    player = {
        chess.WHITE: ChessBot(BOTS[game.headers["White"]]),
        chess.BLACK: ChessBot(BOTS[game.headers["Black"]])
    }

    print("Initial Board:")
    print_board(board)
    i = 0

    while not board.is_game_over():
        print("Player Turn:", "WHITE" if board.turn else "BLACK")
        board.push(player[board.turn].generate_move(board, 1))
        print_board(board)
        node = node.add_variation(board.peek())
        i = (i + 1) % 10
        if not i:
            print(threading.active_count(), "threads active.")

    if board.is_checkmate():
        print('Checkmate!', "Player", "WHITE" if not board.turn else "BLACK", "wins!")
    elif board.is_insufficient_material():
        print('Draw by insufficient material!')
    elif board.is_stalemate():
        print('Draw by stalemate!')
    elif board.is_seventyfive_moves():
        print('Draw by 75-move rule!')
    elif board.is_fivefold_repetition():
        print('Draw by 5-fold repetition!')
    else:
        print('Unexpected game over!?')
    print("Match lasted for", len(board.move_stack), "turns.")
    print(f"WHITE had {player[chess.WHITE].error_count[0]}/{player[chess.WHITE].error_count[1]} time limit errors and {player[chess.WHITE].error_count[2]} invalid moves.")
    print(f"BLACK had {player[chess.BLACK].error_count[0]}/{player[chess.BLACK].error_count[1]} time limit errors and {player[chess.BLACK].error_count[2]} invalid moves.")

    game.headers["Result"] = board.result()

    with open("matches.pgn", "a+", encoding="utf-8") as f:
        f.write(str(game) + "\n\n")



if __name__ == '__main__':
    main()
