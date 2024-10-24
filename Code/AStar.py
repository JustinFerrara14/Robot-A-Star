import math
import heapq
import time


class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
        self.prev_dir = None  # Stocker la direction précédente


class AStarAlgorithm:
    def __init__(self, map_x, map_y, spacing, obstacles=[]):
        self.ROW = int(map_y / spacing)
        self.COL = int(map_x / spacing)
        self.spacing = spacing
        self.grid = [[1 for _ in range(self.COL)] for _ in range(self.ROW)]

        # Enlever les points près des murs
        for i in range(self.ROW):
            for j in range(self.COL):
                if i < 250 / spacing or j < 250 / spacing or i > self.ROW - 250 / spacing or j > self.COL - 250 / spacing:
                    self.grid[i][j] = 0

        # Ajouter des obstacles (si fournis)
        for obstacle in obstacles:
            x, y = obstacle.get_position()
            radius = obstacle.get_radius()
            radius += 250  # Rayon du robot
            x_grid, y_grid = self.mm_to_grid((x, y))
            radius_grid = int(radius / self.spacing)
            for i in range(-radius_grid, radius_grid + 1):
                for j in range(-radius_grid, radius_grid + 1):
                    if self.is_valid(x_grid + i, y_grid + j):
                        self.grid[x_grid + i][y_grid + j] = 0

    def is_valid(self, row, col):
        return (row >= 0) and (row < self.ROW) and (col >= 0) and (col < self.COL)

    def is_unblocked(self, grid, row, col):
        return grid[row][col] == 1

    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    def calculate_h_value(self, row, col, dest):
        return self.spacing * math.sqrt((dest[1] - col) ** 2 + (dest[0] - row) ** 2)

    def mm_to_grid(self, pos_mm):
        return [int(pos_mm[1] / self.spacing), int(pos_mm[0] / self.spacing)]

    def trace_path(self, cell_details, dest):
        path = []
        row, col = dest
        prev_dir = None  # Variable pour suivre la direction précédente

        # On commence par le point de destination
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j

            # Calculer la direction actuelle
            current_dir = (row - temp_row, col - temp_col)

            # Ajouter le point uniquement si la direction change
            if prev_dir is None or current_dir != prev_dir:
                path.append((col * self.spacing, row * self.spacing))

            # Mettre à jour la direction précédente
            prev_dir = current_dir

            # Se déplacer vers le parent
            row, col = temp_row, temp_col

        # Ajouter le point de départ
        path.append((col * self.spacing, row * self.spacing))

        # Inverser le chemin pour qu'il soit dans l'ordre correct
        path.reverse()

        # Supprimer les doublons éventuels dans le chemin (s'il y a des points consécutifs identiques)
        final_path = []
        for pos in path:
            if len(final_path) == 0 or final_path[-1] != pos:
                final_path.append(pos)

        # Affichage du chemin
        curved_path = self.put_points_curve(final_path, 100)
        for position in curved_path:
            print("->", position, end=" ")
        print()

        return curved_path

    import math

    def put_points_curve(self, path, max_curve_distance):
        curved_path = []

        for i in range(len(path) - 2):
            p1 = path[i]  # Current point
            p2 = path[i + 1]  # Next point
            p3 = path[i + 2]  # Point after p2

            # Calculate vectors between points
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Length of segments between points
            d1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            d2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            # Add the current point p1 to the path
            if i == 0:
                curved_path.append(p1)

            # Compute the distance to start and end the curve
            curve_start_dist = min(max_curve_distance, d1 / 2)
            curve_end_dist = min(max_curve_distance, d2 / 2)

            # Place points for the curve if the distances are sufficient
            if curve_start_dist > 0:
                curve_start = (
                    p2[0] - (v1[0] / d1) * curve_start_dist,
                    p2[1] - (v1[1] / d1) * curve_start_dist,
                )
                curved_path.append(curve_start)

            if curve_end_dist > 0:
                curve_end = (
                    p2[0] + (v2[0] / d2) * curve_end_dist,
                    p2[1] + (v2[1] / d2) * curve_end_dist,
                )
                curved_path.append(curve_end)

        # Add the last two points of the original path
        curved_path.append(path[-1])

        return curved_path

    def a_star_search(self, src, dest):
        src_grid = self.mm_to_grid(src)
        dest_grid = self.mm_to_grid(dest)

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
        cell_details[i][j].prev_dir = (0, 0)  # Direction initiale

        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
        found_dest = False
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
                        cell_details[new_i][new_j].prev_dir = dir  # Sauvegarder la direction finale
                        found_dest = True
                        return self.trace_path(cell_details, dest_grid)
                    else:
                        g_new = cell_details[i][j].g + self.spacing * math.sqrt(dir[0] ** 2 + dir[1] ** 2)

                        # Ajouter une pénalité si la direction change
                        if cell_details[i][j].prev_dir != dir:
                            g_new += self.spacing  # Pénalité pour changement de direction

                        h_new = self.calculate_h_value(new_i, new_j, dest_grid)
                        f_new = g_new + h_new

                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j
                            cell_details[new_i][new_j].prev_dir = dir  # Enregistrer la direction

        if not found_dest:
            print("Failed to find the destination cell")


def main():
    src = [500, 500]
    dest = [2500, 1500]
    map_x = 4000
    map_y = 2000
    spacing = 20

    astar = AStarAlgorithm(map_x, map_y, spacing)
    start = time.time()
    path = astar.a_star_search(src, dest)
    end = time.time()

    if path:
        print("Path:", path)
    print("Time:", end - start, "seconds")


if __name__ == "__main__":
    main()
