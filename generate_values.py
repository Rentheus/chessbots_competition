from stockfish import Stockfish
import chess.pgn
import chess
import csv

stockfish = Stockfish(path="Q:/Lib/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")

pgn = open("lichess_elite_2020-07.pgn")

posval = []

num = 0






for i in range(420):
    game = chess.pgn.read_game(pgn)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        stockfish.set_fen_position(board.fen())
        advantage = stockfish.get_evaluation()
        if advantage["type"] == "mate":
            posval.append([board.fen(), 50*advantage["value"]])
        else:
            posval.append([board.fen(), advantage["value"]/100])
    posval[-1][1] = 20

    if i%12 == 0:
        with open("posval.csv2", "a", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerows(posval)
        posval = []



#with open("posval.csv", "a", newline="") as csvf:
#    writer = csv.writer(csvf)
#    writer.writerows(posval)