from enum import Enum, auto
import pygame as pg
import vi
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize
import numpy as np
from numpy import random as r


@deserialize
@dataclass
class AggregationConfig(Config):
    a: float = 2.6
    b: float = 2.2

    delta_time: float = 1.0

    def weights(self) -> tuple[float, float]:
        return (self.a, self.b)


class Bee(Agent):
    state: int = 0 # 0 -> wandering, 1 -> joining, 2 -> still, 3 -> leaving
    t: int = r.randint(40,50)


    config: Config

    def wandering(self, in_proximity, a):
        self.pos += self.move
        uniform_roll = r.uniform()
        p_join = 0.03 + 0.48 * (1 - np.exp(-a * in_proximity))

        if self.on_site():
            if p_join > uniform_roll:
                print('Agent joining')
                self.state = 1

    def join(self):
        if self.t == 0:
            print('stop movement')
            self.t = 30
            self.state = 2
        else:
            print(self.t)
            self.pos += self.move



    def still(self):
        self.freeze_movement()


    def leave(self, b):
        pass

    def change_position(self):
        in_proximity = self.in_proximity_accuracy().count()
        a, b = self.config.weights()
        self.there_is_no_escape()

        if self.state == 0:
            self.wandering(in_proximity, a)
        if self.state == 1:
            self.t -= 1
            self.join()
        if self.state == 2:
            self.still()
        if self.state == 3:
            self.leave(b)



(
    Simulation(
        AggregationConfig(
            image_rotation=True,
            movement_speed=1.5,
            radius=30,
            seed=2,
            fps_limit=60,
        )
    )
        .spawn_site("bubble-full.png", 375, 375)
        .batch_spawn_agents(50, Bee, images=["green.png"])
        .run()
)
