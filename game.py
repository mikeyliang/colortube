import copy
from tubes import Tubes
import cv2

MAX_HEIGHT = 4
class Game(Tubes):
    
    def __init__(self, img):
        super().__init__(img)
        self.colors = self.getGameColors() # List of colors in the current game level
        self.tubes = self.getTubeColors() # Left -> Tube Bottom, Right -> Tube Top
        print("\nGAME LOADED!")
        # self.displayImg()
        self.plot_tubes()

    def convertToString(self, tubes):
        result = []
        for t in self.tubes:
            new_tube = []
            for c in t:
                new_tube.append(str(c))
            result.append(new_tube)
        self.tubes = result

    def solve(self):
        self.convertToString(self.tubes)

        for i, t in enumerate(self.tubes):
            if (i < len(self.tubes) - 1):
                answer = []

                new_game = copy.copy(self.tubes)
                self.solve_game(new_game, [], answer)
            
                # answers are pushed in reverse order, so reverse it to get answer from beginning
                answer.reverse()

                if len(answer) > 0:
                    for a in answer:
                        print(a)

                    break
                else:
                    first_tube = self.tubes.pop()
                    self.tubes.insert(0, first_tube)
                    print(self.tubes)
        
        

    def is_solved(self, tubes):
        for t in tubes:
            if (len(t) > 0):
                if (len(t) < MAX_HEIGHT):
                    return False
                if (self.count_color_in_tube(t, t[0]) != MAX_HEIGHT):
                    return False
        return True
                
    def count_color_in_tube(tube, color):
        max = 0
        for c in tube:
            if (c == color):
                max += 1

        return max

    def can_move(self, source, destination):
        if len(source) == 0 or len(destination) == MAX_HEIGHT:
            return False
        color_in_tube = self.count_color_in_tube(source, source[0])

        # dont move cuz same color
        if (color_in_tube == MAX_HEIGHT):
            return False
        
        if (len(destination) == 0):
            # dont move cuz same color in destination
            if (color_in_tube == len(source)):
                return False
                
            # move cuz empty
            return True
        return source[len(source) - 1] == destination[len(destination) - 1]

    # must be valid pour
    def pour(self, source, destination):
        # always move one
        top = source.pop()
        destination.append(top)
        while len(source) > 0:
            # look at next and compare
            next = source[len(source) - 1]
            if (next == top and self.can_move(self, source, destination)):
                destination.append(source.pop())
            else:
                break
            
    def get_game_as_string(self, game):
        result = ''
        for t in game:
            for c in t:
                result += c
        return result

    def get_tube_as_string(tube):
        result = ''
        for c in tube:
            result += c
        return result

    def solve_game(self, current_game, visited_tubes, answer):
        visited_tubes.append(self.get_game_as_string(current_game))
        for i, t1 in enumerate(current_game):
            for j, t2 in enumerate(current_game):
                if (i == j):
                    continue
                if (self.can_move(t1, t2)):
                    new_game = copy.deepcopy(current_game)

                    # new_game[j].append(new_game[i].pop())
                    self.pour(self, new_game[i], new_game[j])

                    if (self.is_solved(new_game)):
                        answer.append([i, j])
                        print("SOLVED::")
                        print(new_game)
                        return True
                    # recursively try to solve next iteration
                    if (self.get_game_as_string(new_game) not in visited_tubes):
                        solve_next = self.solve_game(self, new_game, visited_tubes, answer)
                        if (solve_next):
                            answer.append([i, j])
                            return True
        return answer
            