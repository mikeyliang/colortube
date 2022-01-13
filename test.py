import cv2 

from game import TubeGame

im = cv2.imread(r"images/colors.jpeg")

game = TubeGame(im)

game.displayGame()