import numpy as np
import math
from gymnasium import Env, spaces
from Environment.Obstacle import Obstacle


class Environment(Env):
    def __init__(self, map_max_x, map_max_y, goal_position, pos_init_robot, angle_init_robot, obstacles=None):
        # Convert tuple in numpy array
        # self.save_pos_init_robot = np.array(pos_init_robot)
        # self.save_angle_init_robot = np.array(angle_init_robot)

        self.save_pos_init_robot = pos_init_robot
        self.save_angle_init_robot = angle_init_robot

        self.map_max_x = map_max_x  # Largeur de la carte 4000
        self.map_max_y = map_max_y  # Hauteur de la carte 2000
        self.map_max_dist = np.linalg.norm([self.map_max_x, self.map_max_y])
        self.robot_positions = np.zeros((1, 2))  # positions des robots (x, y)
        self.robot_init_pos = np.zeros((1, 2))
        self.robot_velocities = np.zeros(1)  # vitesses des robots
        self.robot_angles = np.zeros(1)  # angle des robots
        self.threshold_distance = 20  # Distance seuil pour détecter une collision
        self.goal_position = goal_position  # Position de l'objectif
        self.iteration_count = 0
        self.robot_diameter = 300
        self.distance_moved = 0  # Distance totale parcourue par le robot
        self.time_interval = 0.01  # Temps entre chaque appel d'un update de position en s
        self.time_interval_command = 0.2  # Temps entre chaque modification possible par réseau neuronnes en s
        self.max_speed = 50  # Vitesse max du robot en mm/s
        self.count_step_randomRobots = 0
        self.robot_last_positions = np.zeros((1, 2))
        self.robot_last_angles = np.zeros(1)
        self.max_percentage_deviation = 6  # 1.5 -> 150% des steps minimum pour atteindre la cible
        self.max_iterations = self.init_max_iterations()
        # self.initialize_robot_positions(self.save_pos_init_robot, self.save_angle_init_robot)
        self.obstacles = obstacles
        self.init_goal_angle = 0

        # Définir les espaces d'action et d'observation
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,),
                                       dtype=np.float32)  # 2 valeurs entre -1 et 1 pour sorties
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # 4 valeurs en entrée

    def init_max_iterations(self):
        self.max_iterations = int(
            self.max_percentage_deviation * np.linalg.norm(self.goal_position - self.robot_positions[0]) / (
                    self.max_speed * self.time_interval_command))

        return self.max_iterations

    def init_robot_positions(self, pos_init_robot=None, angle_init_robot=None, pos_goal=None):

        # Init Robot position
        self.robot_positions = np.zeros((1, 2))
        if pos_init_robot is not None:
            self.robot_positions[0] = pos_init_robot
        else:
            self.robot_positions[0] = np.array([np.random.randint(151, self.map_max_x - 151),
                                                np.random.randint(151, self.map_max_y - 151)])

        self.robot_init_pos = np.copy(self.robot_positions[0])

        # Init Goal position
        if pos_goal is not None:
            self.goal_position = pos_goal
        else:
            self.goal_position = np.array([np.random.randint(151, self.map_max_x - 151),
                                           np.random.randint(151, self.map_max_y - 151)])

        # Init angle
        if angle_init_robot is not None:
            self.robot_angles = np.array([angle_init_robot])
        else:
            self.robot_angles = np.array([np.random.randint(0, 360)])

        self.save_angle_init_robot = np.copy(self.robot_angles[0])
        self.iteration_count = 0
        self.init_goal_angle = self.angleGoal()

    def init_obstacles(self):
        if self.obstacles is None:
            return

        obstacles = []
        # Mettre un obstacle autour du goal
        x = np.random.randint(300, 500)
        y = np.random.randint(300, 500)
        coeff_x = np.random.choice([-1, 1])
        coeff_y = np.random.choice([-1, 1])
        obstacles.append(Obstacle(self.goal_position[0] + x * coeff_x, self.goal_position[1] + y * coeff_y, 100))

        self.obstacles = obstacles

    def check_collision(self):
        robot_radius = self.robot_diameter / 2
        if self.robot_positions[0][0] + robot_radius > self.map_max_x:
            return True
        elif self.robot_positions[0][0] - robot_radius < 0:
            return True
        elif self.robot_positions[0][1] + robot_radius > self.map_max_y:
            return True
        elif self.robot_positions[0][1] - robot_radius < 0:
            return True

        if self.obstacles is not None:
            for obstacle in self.obstacles:
                if np.linalg.norm(self.robot_positions[
                                      0] - obstacle.get_position()) < self.robot_diameter / 2 + obstacle.get_radius():
                    return True

        return False

    def update_positions(self, robot_index, wheel_speeds):
        new_pos_x = self.robot_positions[robot_index][0]
        new_pos_y = self.robot_positions[robot_index][1]
        act_angle_rad = self.robot_angles[robot_index] * (math.pi / 180)

        self.robot_last_angles = np.copy(self.robot_angles[0])

        v_left, v_right = wheel_speeds
        v_right *= self.max_speed
        v_left *= self.max_speed

        dist_right = v_right * self.time_interval
        dist_left = v_left * self.time_interval

        delta_distance = (dist_right + dist_left) / 2
        delta_angle = math.atan((dist_left - dist_right) / self.robot_diameter)

        self.distance_moved += delta_distance

        new_pos_x += delta_distance * math.sin(act_angle_rad * -1)
        new_pos_y += delta_distance * math.cos(act_angle_rad)
        act_angle_rad += delta_angle

        if act_angle_rad < 0:
            act_angle_rad += 2 * math.pi
        elif act_angle_rad >= (2 * math.pi):
            act_angle_rad -= 2 * math.pi

        self.robot_angles[robot_index] = act_angle_rad * (180 / math.pi)
        self.robot_positions[robot_index][0] = new_pos_x
        self.robot_positions[robot_index][1] = new_pos_y
        self.robot_velocities[robot_index] = (v_left + v_right) / 2

        return self.robot_positions


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.init_robot_positions()
        self.init_max_iterations()

        self.init_obstacles()

        return self.getState(), {}


    def step(self, actions):
        self.robot_last_positions = np.copy(self.robot_positions)

        for _ in range(0, int(self.time_interval_command / self.time_interval)):
            self.update_positions(0, actions)
            if self.check_collision():
                break

        self.iteration_count += 1

        done = False
        if self.check_collision():  # Collision
            done = True
        elif np.linalg.norm(self.robot_positions[0] - self.goal_position) < self.threshold_distance: # Goal reached
            done = True

        if False:
            print("Terminated : ")
            print("Goal pos: ", self.goal_position)
            print("Init pos: ", self.robot_init_pos)
            print("Init angle: ", self.save_angle_init_robot)
            print("Final pos: ", self.robot_positions[0])
            for obs in self.obstacles:
                print("Obstacle pos: ", obs.get_position())

        return done

    def pilot_robot(self, goal):
        # Angle entre la direction actuelle du robot et l'objectif
        angle_to_goal = self.anglePosition(goal)

        # Paramètres de contrôle
        turn_threshold = 10  # Seuil pour décider si le robot doit tourner
        turn_speed = 0.6  # Vitesse de rotation
        forward_speed = 1  # Vitesse en ligne droite

        # Initialisation des vitesses de roues
        v_left = 0
        v_right = 0

        # Cas 1 : le robot doit tourner pour aligner sa direction avec l'objectif
        if abs(angle_to_goal) > turn_threshold:
            # Si l'angle est positif, tourner à droite (vitesse roue gauche > vitesse roue droite)
            if angle_to_goal > 0:
                v_left = turn_speed
                v_right = -turn_speed
            # Si l'angle est négatif, tourner à gauche (vitesse roue droite > vitesse roue gauche)
            else:
                v_left = -turn_speed
                v_right = turn_speed

        # Cas 2 : le robot est presque aligné avec l'objectif, avancer en ligne droite
        else:
            v_left = forward_speed
            v_right = forward_speed

        # Normaliser les vitesses entre -1 et 1
        v_left = np.clip(v_left, -1, 1)
        v_right = np.clip(v_right, -1, 1)

        return np.array([v_left, v_right])

    def normalizePositions(self):  # TODO assert
        result = []
        for i, value in enumerate(self.goal_position.flatten()):  # TODO check goal position ??
            if i % 2 == 0:  # Si l'indice est pair
                result.append(value / self.map_max_x)
            else:
                result.append(value / self.map_max_y)
        return result

    def normalizeVelocities(self):
        norm_velocities = self.robot_velocities / self.max_speed
        assert -1 <= norm_velocities <= 1
        return norm_velocities

    def normalizeAngles(self):
        norm_angles = (self.robot_angles - 180) / 180
        assert -1 <= norm_angles <= 1
        return norm_angles

    def distanceGoal(self):
        dist = np.linalg.norm(self.goal_position - self.robot_positions[0])
        assert 0 <= dist <= self.map_max_dist
        return dist

    def normalizeDistanceGoal(self):
        dist_goal = self.distanceGoal() / self.map_max_dist
        dist_goal = 2 * dist_goal - 1
        assert -1 <= dist_goal <= 1
        return np.array([dist_goal])

    def angleGoal(self):  # Angle entre goal et direction du robot, gauche -> négatif, droite -> positif
        # Calculer la direction du goal par rapport à la position du robot
        goal_direction = self.goal_position - self.robot_positions[0]

        # Calculer l'angle entre le robot et le goal
        angle_to_goal = math.degrees(math.atan2(goal_direction[1], goal_direction[0]))

        # Calculer l'angle relatif en fonction de l'orientation du robot
        angle_relative = angle_to_goal - self.robot_angles[0]
        angle_relative += 270

        # Normaliser l'angle relatif entre -180 et 180 degrés
        if angle_relative < -180:
            angle_relative += 360
        elif angle_relative > 180:
            angle_relative -= 360

        assert -180 <= angle_relative <= 180
        return angle_relative

    def anglePosition(self, pos): # Angle entre position et direction du robot, gauche -> négatif, droite -> positif
        # Calculer la direction du goal par rapport à la position du robot
        goal_direction = pos - self.robot_positions[0]

        # Calculer l'angle entre le robot et le goal
        angle_to_goal = math.degrees(math.atan2(goal_direction[1], goal_direction[0]))

        # Calculer l'angle relatif en fonction de l'orientation du robot
        angle_relative = angle_to_goal - self.robot_angles[0]
        angle_relative += 270

        # Normaliser l'angle relatif entre -180 et 180 degrés
        if angle_relative < -180:
            angle_relative += 360
        elif angle_relative > 180:
            angle_relative -= 360

        assert -180 <= angle_relative <= 180
        return angle_relative

    def normalizeAngleGoal(self):  # Angle entre goal et direction du robot
        angle_goal = self.angleGoal() / 180
        assert -1 <= angle_goal <= 1
        return np.array([angle_goal])

    def distanceWall(self):
        # return the distance to the closest wall
        dist = np.min([
            self.robot_positions[0][0] - self.robot_diameter / 2,
            self.map_max_x - self.robot_positions[0][0] - self.robot_diameter / 2,
            self.robot_positions[0][1] - self.robot_diameter / 2,
            self.map_max_y - self.robot_positions[0][1] - self.robot_diameter / 2
        ])

        if dist < 0:
            dist = 0

        assert 0 <= dist <= self.map_max_dist
        return dist

    def normalizeDistanceWall(self):
        dist = self.distanceWall() / self.map_max_dist
        dist = 2 * dist - 1
        assert -1 <= dist <= 1
        return np.array([dist])

    def angleWall(self):
        # Get the robot's position and angle
        x, y = self.robot_positions[0]
        robot_angle = self.robot_angles[0]

        # Calculate the angles to each of the walls
        angle_to_left_wall = 90
        angle_to_right_wall = -90
        angle_to_top_wall = 180
        angle_to_bottom_wall = 0

        # Calculate distances to each wall
        distance_to_left_wall = x
        distance_to_right_wall = self.map_max_x - x
        distance_to_top_wall = y
        distance_to_bottom_wall = self.map_max_y - y

        # Find the closest wall
        distances = [distance_to_left_wall, distance_to_right_wall, distance_to_top_wall, distance_to_bottom_wall]
        angles = [angle_to_left_wall, angle_to_right_wall, angle_to_top_wall, angle_to_bottom_wall]
        closest_wall_index = np.argmin(distances)
        angle_to_closest_wall = angles[closest_wall_index]

        # Calculate the relative angle to the closest wall
        relative_angle_to_wall = angle_to_closest_wall - robot_angle
        if relative_angle_to_wall < -180:
            relative_angle_to_wall += 360
        elif relative_angle_to_wall > 180:
            relative_angle_to_wall -= 360

        assert -180 <= relative_angle_to_wall <= 180
        return relative_angle_to_wall

    def normalizeAngleWall(self):
        angle_wall = self.angleWall()
        angle_wall /= 180
        assert -1 <= angle_wall <= 1
        return np.array([angle_wall])

    def distanceObstacle(self):
        # return the distance to the closest obstacle or wall
        if self.obstacles is None:
            return self.distanceWall()

        dist = 1000000000
        for obstacle in self.obstacles:
            dist = min(dist, np.linalg.norm(
                self.robot_positions[0] - obstacle.get_position()) - obstacle.get_radius() - self.robot_diameter / 2)
        if self.distanceWall() < dist:
            dist = self.distanceWall()

        if dist < 0:
            dist = 0

        assert 0 <= dist <= self.map_max_dist
        return dist

    def normalizeDistanceObstacle(self):
        dist = self.distanceObstacle() / self.map_max_dist
        dist = 2 * dist - 1
        assert -1 <= dist <= 1
        return np.array([dist])

    def angleObstacle(self):
        if self.obstacles is None:
            return self.angleWall()

        x, y = self.robot_positions[0]
        robot_angle = self.robot_angles[0]

        dist = 1000000000
        angle = 0
        for obstacle in self.obstacles:
            dist_obstacle = np.linalg.norm(self.robot_positions[0] - obstacle.get_position()) - obstacle.get_radius()
            if dist_obstacle < dist:
                dist = dist_obstacle
                angle = math.degrees(math.atan2(obstacle.get_position()[1] - y, obstacle.get_position()[0] - x))

        angle_relative = angle - robot_angle
        angle_relative += 270

        if dist > self.distanceWall():
            angle_relative = self.angleWall()

        if angle_relative < -180:
            angle_relative += 360
        elif angle_relative > 180:
            angle_relative -= 360

        assert -180 <= angle_relative <= 180
        return angle_relative

    def normalizeAngleObstacle(self):
        angle_obstacle = self.angleObstacle() / 180
        assert -1 <= angle_obstacle <= 1
        return np.array([angle_obstacle])
