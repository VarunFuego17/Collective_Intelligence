from enum import Enum, auto
import pygame as pg
import vi
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize




class Bee(Agent):
    state = 0
    config: Config


    def wandering(self):
        pass
    def still(self):
        pass
    def join(self):
        pass
    def leave(self):
        pass


    def change_position(self):
        self.there_is_no_escape()
        self.pos += self.move
        self.wandering()

        in_proximity = self.in_proximity_accuracy().count()

        if self.on_site():
            self.freeze_movement()
            
            
(
    Simulation(
        Config(
            image_rotation=True,
            movement_speed=1.0,
            radius=15,
            seed=1,
            fps_limit =60,
        )
    )
        .spawn_site("site_medium.png", 375, 375)
        .batch_spawn_agents(30, Bee, images=["bees.png"])
        .run()
)

