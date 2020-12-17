import sys
import numpy as np
from numpy import sin, cos
import scipy.integrate
import pygame
import pygame.gfxdraw

class Point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def translate(self, dx, dy):
		self.x = self.x + dx
		self.y = self.y + dy

		return self

	def scale(self, fx, fy):
		self.x = self.x * fx
		self.y = self.y * fy

		return self

	def round(self):
		self.x = int(self.x)
		self.y = int(self.y)

		return self

	def to_tuple(self):
		return (self.x, self.y)

	def __repr__(self):
		return "{0}, {1}".format(self.x, self.y)

## Parameters of the double pendulum

# Length, in meters, of the two rods
l1 = l2 = 0.3
# Masses, in kilograms, of the two spheres
m1 = 0.5
m2 = 0.5
# Gravitational constant
g = 9.81

def pendulum_ODE_system(t, w):
	"""
	Defines the differential equations for the double pendulum system
	
	Reference for the differential equations:
	https://www.myphysicslab.com/pendulum/double-pendulum-en.html
	
	Arguments:
		t : time
		w :  vector of the state variables:
			w = [y1, y2, y3, y4]
	"""

	y1, y2, y3, y4 = w

	f = [y2,
		(-g*(2*m1+m2)*sin(y1) - m2*g*sin(y1 - 2*y3) - 2*sin(y1-y3)*m2*(y4**2*l2+y1**2*l1*cos(y1-y3)))/(l1*(2*m1 + m2 - m2*cos(2*y1 - 2*y3))),
		y4,
		(2*sin(y1-y3)*(y2**2*l1*(m1+m2)+g*(m1+m2)*cos(y1)+y4**2*l2*m2*cos(y1-y3)))/(l2*(2*m1+m2-m2*cos(2*y1-2*y3)))]

	return f

def draw_aa_circle(surface, center, radius, color):
	x, y = center
	pygame.gfxdraw.aacircle(surface, x, y, radius, color)
	pygame.gfxdraw.filled_circle(surface, x, y, radius, color)

## Initial conditions
# y1: angle (rad), first pendulum
# y2: angular velocity (rad/s), first pendulum
# y3: angle (rad), second pendulum
# y4: angular velocity (rad/s), second pendulum
# t0: initial time (in seconds)
y1_0 = np.radians(45)
y2_0 = 2
y3_0 = np.radians(0)
y4_0 = 0
t0 = 0

initial_conditions = [y1_0, y2_0, y3_0, y4_0]

## PyGame

pygame.init()

WIDTH = 1200
HEIGHT = 800
CENTER = Point(WIDTH / 2, 20).round()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

size = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

last_solution = initial_conditions
last_time = t0

TRACING = True
TRACING_MODE = "lines" # "lines" or "dots"
TRACING_MAX_DOTS = np.inf
pendulum_2_position_history = []

i = 0

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.display.quit()
			sys.exit()

	# dt: the time (ms) since the last frame
	# We divide by 1000 to get seconds
	dt = clock.tick(50) / 1000

	new_solution = scipy.integrate.solve_ivp(pendulum_ODE_system, (last_time, last_time + dt), last_solution)

	index = len(new_solution.t) - 1
	y1 = new_solution.y[0, index]
	y2 = new_solution.y[1, index]
	y3 = new_solution.y[2, index]
	y4 = new_solution.y[3, index]

	last_solution = new_solution.y[:, index]
	last_time = new_solution.t.max()

	pendulum_1 = Point(l1 * sin(y1), - l1 * cos(y1))
	pendulum_2 = Point(pendulum_1.x + l2 * sin(y3), pendulum_1.y - l2 * cos(y3))

	pendulum_1.scale(1000, -1000).translate(WIDTH / 2, CENTER.y).round()
	pendulum_2.scale(1000, -1000).translate(WIDTH / 2, CENTER.y).round()

	screen.fill(WHITE)

	if TRACING:
		if TRACING_MODE == "lines":
			pendulum_2_position_history.append(pendulum_2)

			for idx, position in enumerate(pendulum_2_position_history):
				if idx > 0:
					previous_point = pendulum_2_position_history[idx - 1]
					pygame.draw.aaline(screen, BLUE, previous_point.to_tuple(), position.to_tuple(), 5)
		else:
			if i < TRACING_MAX_DOTS:
				pendulum_2_position_history.append(pendulum_2)
			else:
				pendulum_2_position_history[i % TRACING_MAX_DOTS] = pendulum_2

			i = i + 1

			for position in pendulum_2_position_history:
				draw_aa_circle(screen, position.to_tuple(), 4, BLUE)

	pygame.draw.aaline(screen, BLACK, CENTER.to_tuple(), pendulum_1.to_tuple(), 5)
	pygame.draw.aaline(screen, BLACK, pendulum_1.to_tuple(), pendulum_2.to_tuple(), 5)
	draw_aa_circle(screen, CENTER.to_tuple(), 15, BLACK)
	draw_aa_circle(screen, pendulum_1.to_tuple(), 15, GREEN)
	draw_aa_circle(screen, pendulum_2.to_tuple(), 15, BLUE)

	pygame.display.flip()
