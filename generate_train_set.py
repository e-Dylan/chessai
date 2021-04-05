import chess.pgn
import os
import numpy as np
from state import State

pgn = open("data/games.pgn")

def get_training_data(num_samples=1000):
    
    X, y = [], []
    game_number = 0
    values = {"1-0":1, "0-1":-1, "1/2-1/2":0}
        
    while 1:
        try:
            game = chess.pgn.read_game(pgn)
        except Exception:
            # Ran out of games in data 
            break

        print(f"parsing game: {game_number}, total movestates {len(X)}")

        res = game.headers["Result"]
        if res not in values:
            continue
        value = values[res]
        board = game.board()

        for i, move in enumerate(game.mainline_moves()):
            # all moves in a game
            board.push(move)
            ser = State(board).serialize()
            X.append(ser)
            y.append(value)
            # print(value, serialization)

        if num_samples is not None and len(X) > num_samples:
            return X, y

        game_number += 1

    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    X, y = get_training_data(2000000)
    np.savez('data_processed/train_set_2m.npz', X, y)

