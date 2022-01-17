import cv2 

from tubes import Tubes
from game import Game

im = cv2.imread(r"images/charging.jpeg")

game = Game(im, 'phone')
game.solve()
