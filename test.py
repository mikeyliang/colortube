import cv2 

from tubes import Tubes
from game import Game

im = cv2.imread(r"images/127.jpeg")

game = Game(im, 'screenshot')
game.solve()
