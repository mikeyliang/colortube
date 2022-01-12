import cv2 

from game import TubeGame

im = cv2.imread(r"images/charging.jpeg")

game = TubeGame(im)

game.displayTube(0)