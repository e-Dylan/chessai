from flask import Flask, Response
import time
import chess
import chess.svg
import chessdotcom
import requests
import json

app = Flask(__name__)

@app.route('/')
def game():
    return f"<html><body><img width=900 height=900 src='/board.svg?{time.time()}'>"    

@app.route('/board.svg')
def board():
    board = chess.Board()
    return Response(chess.svg.board(board=board), mimetype="image/svg+xml")

if __name__ == "__main__":
    # app.run(debug=True)

