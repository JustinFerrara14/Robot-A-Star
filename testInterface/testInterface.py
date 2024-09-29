import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import random


from Environment.Obstacle import Obstacle


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Navigation")
        self.root.geometry("1000x800")

        self.environment = None
        self.model = None

        self.model_button = tk.Button(root, text="Select Model", command=self.load_model)
        self.model_button.pack()

        self.start_label = tk.Label(root, text="Enter Start Position (x, y):")
        self.start_label.pack()
        self.start_x = tk.Entry(root)
        self.start_x.pack()
        self.start_y = tk.Entry(root)
        self.start_y.pack()

        self.goal_label = tk.Label(root, text="Enter Goal Position (x, y):")
        self.goal_label.pack()
        self.goal_x = tk.Entry(root)
        self.goal_x.pack()
        self.goal_y = tk.Entry(root)
        self.goal_y.pack()

        self.angle_label = tk.Label(root, text="Enter Initial Angle:")
        self.angle_label.pack()
        self.angle = tk.Entry(root)
        self.angle.pack()

        self.random_button = tk.Button(root, text="Randomize Values", command=self.randomize_values)
        self.random_button.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start_simulation)
        self.start_button.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_gui)
        self.reset_button.pack()

        self.canvas = tk.Canvas(root, width=800, height=400, bg="white")
        self.canvas.pack()

        # Information about the robot
        self.current_pos_x_label = tk.Label(root, text="Position x: 0")
        self.current_pos_x_label.pack()
        self.current_pos_x_label.place(x=10, y=10)

        self.current_pos_y_label = tk.Label(root, text="Position y: 0")
        self.current_pos_y_label.pack()
        self.current_pos_y_label.place(x=10, y=40)

        self.current_angle_label = tk.Label(root, text="Angle: 0°")
        self.current_angle_label.pack()
        self.current_angle_label.place(x=10, y=70)

        self.current_angle_goal_label = tk.Label(root, text="Angle to goal: 0°")
        self.current_angle_goal_label.pack()
        self.current_angle_goal_label.place(x=10, y=100)

        self.current_dist_goal_label = tk.Label(root, text="Dist to goal: 0")
        self.current_dist_goal_label.pack()
        self.current_dist_goal_label.place(x=10, y=130)

        self.current_command_left_label = tk.Label(root, text="Command to left: 0")
        self.current_command_left_label.pack()
        self.current_command_left_label.place(x=10, y=160)

        self.current_command_right_label = tk.Label(root, text="Command to right: 0")
        self.current_command_right_label.pack()
        self.current_command_right_label.place(x=10, y=190)

        # Deuxième colonne
        self.current_angle_obstacle_label = tk.Label(root, text="Angle to obstacle: 0°")
        self.current_angle_obstacle_label.pack()
        self.current_angle_obstacle_label.place(x=150, y=10)

        self.current_dist_obstacle_label = tk.Label(root, text="Dist to obstacle: 0")
        self.current_dist_obstacle_label.pack()
        self.current_dist_obstacle_label.place(x=150, y=40)

        self.current_reward_label = tk.Label(root, text="Reward: 0")
        self.current_reward_label.pack()
        self.current_reward_label.place(x=150, y=70)

        # Step by step mode
        self.step_by_step_mode = tk.BooleanVar(value=False)  # Flag to keep track of the mode
        self.continue_simulation = False  # Flag to continue or pause simulation

        self.step_by_step_label = tk.Label(root, text="Step-by-step Mode")
        self.step_by_step_label.pack()
        self.step_by_step_label.place(x=600, y=10)

        self.step_by_step_slider = ttk.Checkbutton(root, variable=self.step_by_step_mode, onvalue=True, offvalue=False)
        self.step_by_step_slider.pack()
        self.step_by_step_slider.place(x=750, y=10)

        # Pilot mode
        self.manual_command_label = tk.Label(root, text="Enter manual command (left, right):")
        self.manual_command_label.pack()
        self.manual_command_label.place(x=600, y=40)
        self.command_left = tk.Entry(root)
        self.command_left.pack()
        self.command_left.place(x=640, y=60)
        self.command_right = tk.Entry(root)
        self.command_right.pack()
        self.command_right.place(x=640, y=80)

        # Obstacle 1
        self.obstacle1_label = tk.Label(root, text="Obstacle 1 (x, y, radius):")
        self.obstacle1_label.pack()
        self.obstacle1_label.place(x=800, y=0)
        self.obstacle1_x = tk.Entry(root)
        self.obstacle1_x.pack()
        self.obstacle1_x.place(x=800, y=20)
        self.obstacle1_y = tk.Entry(root)
        self.obstacle1_y.pack()
        self.obstacle1_y.place(x=800, y=40)
        self.obstacle1_rad = tk.Entry(root)
        self.obstacle1_rad.pack()
        self.obstacle1_rad.place(x=800, y=60)

        # Obstacle 2
        self.obstacle2_label = tk.Label(root, text="Obstacle 2 (x, y, radius):")
        self.obstacle2_label.pack()
        self.obstacle2_label.place(x=800, y=80)
        self.obstacle2_x = tk.Entry(root)
        self.obstacle2_x.pack()
        self.obstacle2_x.place(x=800, y=100)
        self.obstacle2_y = tk.Entry(root)
        self.obstacle2_y.pack()
        self.obstacle2_y.place(x=800, y=120)
        self.obstacle2_rad = tk.Entry(root)
        self.obstacle2_rad.pack()
        self.obstacle2_rad.place(x=800, y=140)

        # Obstacle 3
        self.obstacle3_label = tk.Label(root, text="Obstacle 3 (x, y, radius):")
        self.obstacle3_label.pack()
        self.obstacle3_label.place(x=800, y=160)
        self.obstacle3_x = tk.Entry(root)
        self.obstacle3_x.pack()
        self.obstacle3_x.place(x=800, y=180)
        self.obstacle3_y = tk.Entry(root)
        self.obstacle3_y.pack()
        self.obstacle3_y.place(x=800, y=200)
        self.obstacle3_rad = tk.Entry(root)
        self.obstacle3_rad.pack()
        self.obstacle3_rad.place(x=800, y=220)

        self.draw_legends()

    """
    def load_model(self):
        model_path = filedialog.askopenfilename()
        self.model = tf.keras.models.load_model(model_path)
    """

    def load_model(self):
        model_path = filedialog.askopenfilename()
        self.model = PPO.load(model_path)

    def reset_gui(self):
        self.reset_environment()
        self.canvas.delete("all")

    def reset_environment(self):
        self.environment = None
        self.continue_simulation = False

    def draw_legends(self):
        # Adding legends for the map
        self.canvas.create_text(10, 10, anchor="nw", text="(0, 0)", fill="black")  # Top-left corner
        self.canvas.create_line(10, 10, 60, 10, arrow=tk.LAST, fill="black")  # X direction arrow
        self.canvas.create_text(70, 10, anchor="nw", text="x", fill="black")  # X label
        self.canvas.create_line(10, 10, 10, 60, arrow=tk.LAST, fill="black")  # Y direction arrow
        self.canvas.create_text(10, 70, anchor="nw", text="y", fill="black")  # Y label

    def draw_robot(self):
        self.canvas.delete("all")

        start_pos = self.environment.robot_positions[0]
        goal_pos = self.environment.goal_position

        scale_x = 800 / self.environment.map_max_x
        scale_y = 400 / self.environment.map_max_y

        robot_radius = self.environment.robot_diameter / 2 * scale_x

        scaled_start_pos = (start_pos[0] * scale_x, start_pos[1] * scale_y)
        scaled_goal_pos = (goal_pos[0] * scale_x, goal_pos[1] * scale_y)

        # Obstacle
        if self.environment.obstacles:
            for obstacle in self.environment.obstacles:
                obstacle_pos = obstacle.get_position()
                obstacle_radius = obstacle.get_radius() * scale_x
                scaled_obstacle_pos = (obstacle_pos[0] * scale_x, obstacle_pos[1] * scale_y)
                self.canvas.create_oval(scaled_obstacle_pos[0] - obstacle_radius, scaled_obstacle_pos[1] - obstacle_radius,
                                        scaled_obstacle_pos[0] + obstacle_radius, scaled_obstacle_pos[1] + obstacle_radius,
                                        fill="black")

        # Robot
        self.canvas.create_oval(scaled_start_pos[0] - robot_radius, scaled_start_pos[1] - robot_radius,
                                scaled_start_pos[0] + robot_radius, scaled_start_pos[1] + robot_radius,
                                fill="blue")

        # Angle 0 en bas, sens horaire, robot aiguille rouge
        angle_rad = math.radians(self.environment.robot_angles[0] - 180)
        front_x = scaled_start_pos[0] + robot_radius * math.sin(angle_rad)
        front_y = scaled_start_pos[1] - robot_radius * math.cos(angle_rad)
        self.canvas.create_line(scaled_start_pos[0], scaled_start_pos[1], front_x, front_y, fill="red", width=2)

        # Goal
        goal_radius = 10
        self.canvas.create_oval(scaled_goal_pos[0] - goal_radius, scaled_goal_pos[1] - goal_radius,
                                scaled_goal_pos[0] + goal_radius, scaled_goal_pos[1] + goal_radius,
                                fill="green")

        # Adding legends for the map
        self.draw_legends()

        current_pos_x = self.environment.robot_positions[0][0]
        self.current_pos_x_label.config(text=f"Position x: {current_pos_x:.2f}")

        current_pos_y = self.environment.robot_positions[0][1]
        self.current_pos_y_label.config(text=f"Position y: {current_pos_y:.2f}")

        current_angle = self.environment.robot_angles[0]
        self.current_angle_label.config(text=f"Angle: {current_angle:.2f}°")

        angle_to_goal = self.environment.angleGoal()
        self.current_angle_goal_label.config(text=f"Angle to goal: {angle_to_goal:.2f}°")

        dist_to_goal = self.environment.distanceGoal()
        self.current_dist_goal_label.config(text=f"Dist to goal: {dist_to_goal:.2f}")

        angle_to_obstacle = self.environment.angleObstacle()
        self.current_angle_obstacle_label.config(text=f"Angle to obstacle: {angle_to_obstacle:.2f}°")

        dist_to_obstacle = self.environment.distanceObstacle()
        self.current_dist_obstacle_label.config(text=f"Dist to obstacle: {dist_to_obstacle:.2f}")

    def randomize_values(self):
        max_x = 4000  # Assuming the max value for x
        max_y = 2000  # Assuming the max value for y
        self.start_x.delete(0, tk.END)
        self.start_y.delete(0, tk.END)
        self.goal_x.delete(0, tk.END)
        self.goal_y.delete(0, tk.END)
        self.angle.delete(0, tk.END)
        self.obstacle1_x.delete(0, tk.END)
        self.obstacle1_y.delete(0, tk.END)
        self.obstacle1_rad.delete(0, tk.END)
        self.obstacle2_x.delete(0, tk.END)
        self.obstacle2_y.delete(0, tk.END)
        self.obstacle2_rad.delete(0, tk.END)
        self.obstacle3_x.delete(0, tk.END)
        self.obstacle3_y.delete(0, tk.END)
        self.obstacle3_rad.delete(0, tk.END)

        self.start_x.insert(0, str(random.randint(150, max_x - 150)))
        self.start_y.insert(0, str(random.randint(150, max_y - 150)))
        self.goal_x.insert(0, str(random.randint(150, max_x - 150)))
        self.goal_y.insert(0, str(random.randint(150, max_y - 150)))
        self.angle.insert(0, str(random.uniform(0, 360)))

        self.obstacle1_x.insert(0, str(random.randint(150, max_x - 150)))
        self.obstacle1_y.insert(0, str(random.randint(150, max_y - 150)))
        self.obstacle1_rad.insert(0, str(150))
        self.obstacle2_x.insert(0, str(random.randint(150, max_x - 150)))
        self.obstacle2_y.insert(0, str(random.randint(150, max_y - 150)))
        self.obstacle2_rad.insert(0, str(150))
        self.obstacle3_x.insert(0, str(random.randint(150, max_x - 150)))
        self.obstacle3_y.insert(0, str(random.randint(150, max_y - 150)))
        self.obstacle3_rad.insert(0, str(150))



    """
    def start_simulation(self):
        start_pos = (int(self.start_x.get()), int(self.start_y.get()))
        goal_pos = (int(self.goal_x.get()), int(self.goal_y.get()))
        start_angle = float(self.angle.get())

        self.environment = Environment(4000, 2000, np.array(goal_pos))
        self.environment.initialize_robot_positions(start_pos, start_angle)

        done = False
        while not done:
            self.draw_robot()
            state = self.environment.getState().flatten()
            actions = self.model.predict(np.array([state]))[0]
            state, reward, done = self.environment.step(actions)
            self.root.update()
            self.root.after(50)
    """

    def start_simulation(self):
        # self.reset_environment()  # Reset the environment before starting the simulation

        if not self.environment:  # Check if the environment is not already initialized
            start_pos = (int(self.start_x.get()), int(self.start_y.get()))
            goal_pos = (int(self.goal_x.get()), int(self.goal_y.get()))
            start_angle = float(self.angle.get())
            obstacles = []
            if self.obstacle1_x.get() and self.obstacle1_y.get() and self.obstacle1_rad.get():
                obstacles.append(Obstacle(int(self.obstacle1_x.get()), int(self.obstacle1_y.get()), int(self.obstacle1_rad.get())))

            if self.obstacle2_x.get() and self.obstacle2_y.get() and self.obstacle2_rad.get():
                obstacles.append(Obstacle(int(self.obstacle2_x.get()), int(self.obstacle2_y.get()), int(self.obstacle2_rad.get())))

            if self.obstacle3_x.get() and self.obstacle3_y.get() and self.obstacle3_rad.get():
                obstacles.append(Obstacle(int(self.obstacle3_x.get()), int(self.obstacle3_y.get()), int(self.obstacle3_rad.get())))

            self.environment = CustomEnv(4000, 2000, np.array(goal_pos), start_pos, start_angle, obstacles)
            self.environment.init_robot_positions(start_pos, start_angle, np.array(goal_pos))
            self.environment.init_max_iterations()

        self.vec_env = DummyVecEnv([lambda: self.environment])
        self.continue_simulation = True

        while self.continue_simulation:
            self.draw_robot()
            state = self.environment.getState().flatten()

            if self.command_left.get() and self.command_right.get():
                actions = [float(self.command_left.get()), float(self.command_right.get())]
            else:
                actions, _ = self.model.predict(state)

            command_left = actions[0]
            self.current_command_left_label.config(text=f"Command to left: {command_left:.2f}")

            command_right = actions[1]
            self.current_command_right_label.config(text=f"Command to right: {command_right:.2f}")

            state, reward, done, _, _ = self.environment.step(actions)

            self.current_reward_label.config(text=f"Reward: {reward:.2f}")

            self.root.update()
            self.root.after(50)

            if self.step_by_step_mode.get():
                self.continue_simulation = False

            if done:
                self.continue_simulation = False
                self.reset_environment()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
