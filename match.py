"""
main file to test the custom bots
"""

import multiprocessing
import random
import chess
import chess.pgn
from custom_bot import bad_bot, my_bot, BOTS

class ChessBot:
    def __init__(self, logic_func, name = "Bot") -> None:
        manager = multiprocessing.Manager()
        self.shared = manager.Namespace()
        self.shared.best_move = None
        self.logic_func = logic_func
        self.name = name

    def generate_move(self, board: chess.Board, time_limit: int = 5) -> chess.Move:
        self.shared.best_move = None
        process = multiprocessing.Process(target=self.logic_func, args=(self.shared, board))
        process.start()
        process.join(timeout=time_limit)

        if process.is_alive():
            process.terminate()
            process.join()

        if self.shared.best_move is not None:
            return self.shared.best_move
        print("Player did not make a move in time ... Choosing random move")
        return random.choice(list(board.legal_moves))



def print_board(board):
    print(board)
    print(board.fen())
    print("\n\n")


def main():
    board = chess.Board()

    game = chess.pgn.Game()
    game.headers["Event"] = "CHESSBOT_COMPETITON"
    game.headers["White"] = "my_bot"
    game.headers["Black"] = "my_bot"
    node = game

    player = {
        chess.WHITE: ChessBot(BOTS[game.headers["White"]]),
        chess.BLACK: ChessBot(BOTS[game.headers["Black"]])
    }

    print("Initial Board:")
    print_board(board)

    while not board.is_game_over():
        print("Player Turn:", "WHITE" if board.turn else "BLACK")
        board.push(player[board.turn].generate_move(board.copy()))
        print_board(board)
        node = node.add_variation(board.peek())

    if board.is_checkmate():
        print('Checkmate!', "Player", "WHITE" if not board.turn else "BLACK", "wins!")
    elif board.is_insufficient_material():
        print('Draw by insufficient material!')
    elif board.is_stalemate():
        print('Dray by stalemate!')
    elif board.is_seventyfive_moves():
        print('Dray by 75-move rule!')
    elif board.is_fivefold_repetition():
        print('Dray by 5-fold repetition!')
    else:
        print('Unexpected game over!?')

    game.headers["Result"] = board.result()

    print(game, file=open("test.pgn", "w"), end="\n\n")



if __name__ == '__main__':
    main()
