

class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def get_position(self):
        return self.x, self.y

    def get_radius(self):
        return self.radius

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_radius(self, radius):
        self.radius = radius