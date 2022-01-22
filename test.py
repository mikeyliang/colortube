import cv2 

from tubes import Tubes
from game import Game

im = cv2.imread(r"images/249.jpg")

game = Game(im, 'screenshot')
game.solve()
