from tubes import Tubes

class Game(Tubes):
    
    def __init__(self, img):
        super().__init__(img)
        self.colors = self.getGameColors()
        self.tubes = self.getTubeColors()
        print("\nGAME LOADED!")
        self.displayTube()

    
