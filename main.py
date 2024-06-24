import pygame
import math
import sys
import neat

pygame.init()

# Define Constants
SCALE = 1/2
SCREEN_WIDTH = 1634
SCREEN_HEIGHT = 842
SCALED_SCREEN = pygame.display.set_mode(
    (SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE))
SCREEN = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
TEXT_SCREEN = pygame.surface.Surface((SCREEN_WIDTH, SCREEN_HEIGHT),pygame.SRCALPHA, 32)
FONT = pygame.font.Font('FreeMono.ttf', 25)
DRAWING = SCREEN.copy()
show_debug = True
pressed = True
pygame.display.set_caption('Ai Car')


class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load('car.png')
        self.original_image = pygame.transform.rotozoom(self.original_image, 0, 0.1)
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(590, 670))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.roatation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.roatation_vel
            self.vel_vector.rotate_ip(self.roatation_vel)
        elif self.direction == -1:
            self.angle += self.roatation_vel
            self.vel_vector.rotate_ip(-self.roatation_vel)

        self.image = pygame.transform.rotate(self.original_image,self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
                length += 1
                x = int(self.rect.center[0] +
                        math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.rect.center[1] -
                        math.sin(math.radians(self.angle + radar_angle)) * length)
        except IndexError:
            pass
        
        if show_debug:
            pygame.draw.line(SCREEN, (225, 225, 225, 225), self.rect.center,
                            (x, y), 1)
            pygame.draw.circle(SCREEN, (0, 225, 0, 0), (x, y), 3)

        dist = int(
            math.sqrt(
                math.pow(self.rect.center[0] - x, 2) +
                math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def collision(self):
        length = 40
        collision_point_right = [
            int(self.rect.center[0] +
                math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] -
                math.sin(math.radians(self.angle + 18)) * length)
        ]
        collision_point_left = [
            int(self.rect.center[0] +
                math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] -
                math.sin(math.radians(self.angle - 18)) * length)
        ]
        try:
            coll_right = SCREEN.get_at((collision_point_right))
        except IndexError:
            coll_right = pygame.Color(2, 105, 31,255)
        try:
            coll_left = SCREEN.get_at((collision_point_left))
        except IndexError:
            coll_left = pygame.Color(2, 105, 31,255)

        if coll_right == pygame.Color(2, 105, 31,255) or coll_left == pygame.Color(2, 105, 31, 255):
            self.alive = False

        if show_debug:
            pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
            pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input


def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config):
    global cars, ge, nets, show_debug, pressed, SCALED_SCREEN

    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0


    run = True
    while run:
        text1 = FONT.render("Training...", True, (225, 225, 225))
        text2 = FONT.render(f"Generation: {pop.generation+1}", True, (225, 225, 225))
        text3 = FONT.render(f"Toggle Debug lines with \"H\"", True, (225, 225, 225))
        TEXT_SCREEN.fill((0,0,0,0))
        TEXT_SCREEN.blit(text1, (5,0))
        TEXT_SCREEN.blit(text2, (5,25))
        TEXT_SCREEN.blit(text3, (5,50))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(1)
        keys = pygame.key.get_pressed()
        if not pressed:
            if keys[pygame.K_h]:
                show_debug = not show_debug
                pressed = True
        else:
            if not keys[pygame.K_h]:
                pressed = False

        
        SCREEN.blit(DRAWING, (0, 0))

        if len(cars) == 0: break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        # Update car
        for car in cars:
            car.draw(SCREEN)
            car.update()
        render()
        

def render():
    SCALED_SCREEN.blit(pygame.transform.scale(SCREEN,SCALED_SCREEN.get_rect().size), (0, 0))
    SCALED_SCREEN.blit(pygame.transform.scale(TEXT_SCREEN,SCALED_SCREEN.get_rect().size), (0, 0))
    pygame.display.update()

def draw():
    global DRAWING, SCALED_SCREEN
    SCREEN.fill((2, 105, 31))
    TEXT_SCREEN.fill((0,0,0,0))
    pygame.draw.circle(SCREEN, (87,87,87), (590, 670), 100)
    text1 = FONT.render("Draw a path with mouse.", True, (225, 225, 225))
    text2 = FONT.render("Press \"C\" to continue.", True, (225, 225, 225))
    TEXT_SCREEN.blit(text1, (5,0))
    TEXT_SCREEN.blit(text2, (5,25))

    while True:
        m_pressed = pygame.mouse.get_pressed()[0]
        m_pos = list(pygame.mouse.get_pos())
        m_pos[0] /= SCALE 
        m_pos[1] /= SCALE 
        if m_pressed:
            pygame.draw.circle(SCREEN, (87,87,87), m_pos, 60)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_c]:
            DRAWING = SCREEN.copy()
            TEXT_SCREEN.fill((0,0,0,0))

            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(1)
        render()
 


def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.run(eval_genomes, 50)

if __name__ == '__main__':
    draw()
    run('config.txt')