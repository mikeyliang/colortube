from tubes import Tubes

class Game(Tubes):
    
    def __init__(self, img):
        super().__init__(img)
        self.colors = self.getGameColors() # List of colors in the current game level
        self.tubes = self.getTubeColors() # Left -> Tube Bottom, Right -> Tube Top
        print("\nGAME LOADED!")
        self.displayGame()

    



    

    
