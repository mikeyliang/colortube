import copy

MAX_HEIGHT = 4

# game_1 = [['green', 'blue', 'blue', 'blue'], ['yellow', 'blue', 'red', 'yellow'], ['yellow', 'green', 'green', 'yellow'], ['red', 'red', 'red', 'green'], [], []]
# game_1 = [['blue', 'blue', 'blue', 'blue'], ['yellow', 'yellow', 'yellow', 'yellow'], ['red', 'red', 'red', 'red'], ['green', 'green', 'green', 'green'], []]

# game_1 = [['green', 'pink', 'red', 'purple'],
# ['green', 'orange', 'red', 'midgreen'],
# ['orange', 'green', 'brown', 'grey'],
# ['pink', 'yellow', 'purple', 'pink'],
# ['grey', 'orange', 'violet', 'midgreen'],
# ['violet', 'midgreen', 'yellow', 'blue'],
# ['blue', 'violet', 'lightgreen', 'brown'],
# ['blue', 'red', 'violet', 'grey'],
# ['brown', 'orange', 'lightgreen', 'purple'],
# ['midgreen', 'lightgreen', 'blue', 'pink'],
# ['green', 'yellow', 'yellow', 'grey'],
# ['purple', 'lightgreen', 'red', 'brown'], [], []]

# game_1 = [
#    ['orange', 'red', 'blue', 'blue'],
# ['green', 'pink', 'pink', 'green'],
# ['grey', 'violet', 'orange', 'green'],
# ['pink', 'violet', 'grey', 'red'],
# ['orange', 'grey', 'pink', 'green'],
# ['red', 'red', 'orange', 'blue'],
# ['blue', 'violet', 'violet', 'grey'],
# [],
# []
# ]

game_1 = [
   [1, 2, 3, 3],
[4, 5, 5, 4],
[6, 7, 1, 4],
[5, 7, 6, 2],
[1, 6, 5, 4],
[2, 2, 1, 3],
[3, 7, 7, 6],
[],
[]
]

# game_1 = [[1, 2, 3, 3], [2, 2, 1, 3], [4, 5, 5, 4], [3, 6, 6, 7], [7, 6, 1, 4], [0, 0, 0, 0], [5, 6, 7, 2], [0, 0, 0, 0], [1, 7, 5, 4]]

def is_solved(game):
    for t in game:
        if (len(t) > 0):
            if (len(t) < MAX_HEIGHT):
                return False
            if (count_color_in_tube(t, t[0]) != MAX_HEIGHT):
                return False
    return True
            
def count_color_in_tube(tube, color):
    max = 0
    for c in tube:
        if (c == color):
            max += 1

    return max

def can_move(source, destination):
    if len(source) == 0 or len(destination) == MAX_HEIGHT:
        return False
    color_in_tube = count_color_in_tube(source, source[0])

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
def pour(source, destination):
    # always move one
    top = source.pop()
    destination.append(top)
    while len(source) > 0:
        # look at next and compare
        next = source[len(source) - 1]
        if (next == top and can_move(source, destination)):
            destination.append(source.pop())
        else:
           break
        
def get_game_as_string(game):
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

def solve_game(current_game, visited_tubes, answer):
    visited_tubes.append(get_game_as_string(current_game))
    for i, t1 in enumerate(current_game):
        for j, t2 in enumerate(current_game):
            if (i == j):
                continue
            if (can_move(t1, t2)):
                new_game = copy.deepcopy(current_game)

                # new_game[j].append(new_game[i].pop())
                pour(new_game[i], new_game[j])

                if (is_solved(new_game)):
                    answer.append([i, j])
                    print("SOLVED::")
                    print(new_game)
                    return True
                # recursively try to solve next iteration
                if (get_game_as_string(new_game) not in visited_tubes):
                    solve_next = solve_game(new_game, visited_tubes, answer)
                    if (solve_next):
                        answer.append([i, j])
                        return True
    return answer
        
def convertToString(tubes):
    result = []
    for t in tubes:
        new_tube = []
        for c in t:
            new_tube.append(str(c))
        result.append(new_tube)
    return result

def main():
    # for i, t in enumerate(game_1):
    #     print(t)

    current_game = convertToString(game_1)
    print(current_game)

    for i, t in enumerate(current_game):
        if (i < len(current_game) - 1):
            answer = []

            new_game = copy.copy(current_game)
            solve_game(new_game, [], answer)
        
            # answers are pushed in reverse order, so reverse it to get answer from beginning
            answer.reverse()

            if len(answer) > 0:
                for a in answer:
                    print(a)

                break
            else:
                first_tube = current_game.pop()
                current_game.insert(0, first_tube)
                print(current_game)


main()