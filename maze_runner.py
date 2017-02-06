def maze_runner(n, blocked=None):
    maze = []
    for row in range(n):
        maze.append([])
        for col in range(n):
            if (row, col) == blocked:
                maze[row].append(0)
            else:
                try:
                    maze[row].append(maze[row-1][col] + maze[row][col-1])
                except IndexError:
                    maze[row].append(1)
    return maze[n-1][n-1]
