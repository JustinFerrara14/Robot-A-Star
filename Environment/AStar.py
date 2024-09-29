import math
import heapq
# measure time
import time


# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination


class AStarAlgorithm:
    def __init__(self, map_x, map_y, spacing):
        self.ROW = int(map_y / spacing)  # Number of rows in the grid
        self.COL = int(map_x / spacing)  # Number of columns in the grid
        print("Rows: ", self.ROW)
        print("Columns: ", self.COL)
        self.spacing = spacing  # Spacing between points
        self.grid = [[1 for _ in range(self.COL)] for _ in
                     range(self.ROW)]  # Define the grid (1 for unblocked, 0 for blocked)

    # Check if a cell is valid (within the grid)
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.ROW) and (col >= 0) and (col < self.COL)

    # Check if a cell is unblocked
    def is_unblocked(self, grid, row, col):
        return grid[row][col] == 1

    # Check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    # Calculate the heuristic value of a cell (Euclidean distance to destination)
    def calculate_h_value(self, row, col, dest):
        # Taking spacing into account in the distance calculation
        return self.spacing * math.sqrt((row - dest[0]) ** 2 + (col - dest[1]) ** 2)

    # Convert millimeter positions to grid indices
    def mm_to_grid(self, pos_mm):
        return [int(pos_mm[1] / self.spacing), int(pos_mm[0] / self.spacing)]  # y for rows, x for columns

    # Trace the path from source to destination
    def trace_path(self, cell_details, dest):
        print("The Path is ")
        path = []
        row = dest[0]
        col = dest[1]

        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            # Convert grid indices to real positions
            real_position = (col * self.spacing, row * self.spacing)
            path.append(real_position)
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row = temp_row
            col = temp_col

        # Convert the final position as well
        path.append((col * self.spacing, row * self.spacing))
        path.reverse()

        for position in path:
            print("->", position, end=" ")
        print()

        return path

    # Implement the A* search algorithm
    def a_star_search(self, src, dest):
        # Convert the source and destination from millimeters to grid indices
        src_grid = self.mm_to_grid(src)
        dest_grid = self.mm_to_grid(dest)

        print("Source: ", src_grid)
        print("Destination: ", dest_grid)

        if not self.is_valid(src_grid[0], src_grid[1]) or not self.is_valid(dest_grid[0], dest_grid[1]):
            print("Source or destination is invalid")
            return
        if not self.is_unblocked(self.grid, src_grid[0], src_grid[1]) or not self.is_unblocked(self.grid, dest_grid[0],
                                                                                               dest_grid[1]):
            print("Source or the destination is blocked")
            return
        if self.is_destination(src_grid[0], src_grid[1], dest_grid):
            print("We are already at the destination")
            return

        closed_list = [[False for _ in range(self.COL)] for _ in range(self.ROW)]
        cell_details = [[Cell() for _ in range(self.COL)] for _ in range(self.ROW)]

        i, j = src_grid[0], src_grid[1]
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
        found_dest = False

        # Adjusting the directions for diagonal and straight moves
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_list:
            p = heapq.heappop(open_list)
            i, j = p[1], p[2]
            closed_list[i][j] = True

            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]

                if self.is_valid(new_i, new_j) and self.is_unblocked(self.grid, new_i, new_j) and not \
                        closed_list[new_i][new_j]:
                    if self.is_destination(new_i, new_j, dest_grid):
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        found_dest = True
                        return self.trace_path(cell_details, dest_grid)
                    else:
                        # Adding spacing to g cost
                        g_new = cell_details[i][j].g + self.spacing * math.sqrt(dir[0] ** 2 + dir[1] ** 2)
                        h_new = self.calculate_h_value(new_i, new_j, dest)
                        f_new = g_new + h_new

                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j

        if not found_dest:
            print("Failed to find the destination cell")



# Driver Code
def main():
    # Define the source and destination
    src = [500, 500]
    dest = [2500, 1500]

    # Define map size and spacing
    map_x = 4000  # Map width
    map_y = 2000  # Map height
    spacing = 20  # Spacing between nodes

    # Initialize the AStarAlgorithm with the grid size and spacing
    astar = AStarAlgorithm(map_x, map_y, spacing)

    start = time.time()
    # Run the A* search algorithm
    astar.a_star_search(src, dest)

    end = time.time()
    print("Time: ", end - start, "seconds")

if __name__ == "__main__":
    main()
