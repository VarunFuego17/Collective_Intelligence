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
    t: int =   10

    delta_time: float = 1.0

    def weights(self) -> tuple[float, float, int]:
        return (self.a, self.b, self.t)

class Bee(Agent):
    state = 0 # 0 -> wandering, 1 -> joining, 2 -> still, 3 -> leaving
    t_wait = 0
    config: Config


    def wandering(self):
        
        self.pos += self.move
        
        if self.on_site():
            a, _, t = self.config.weights()
            in_proximity = self.in_proximity_accuracy().count()
            uniform_roll = r.uniform()
            
            p_join = 0.03 + 0.48*(1 - np.exp(-a*in_proximity))
            
            if p_join > uniform_roll:
                self.state = 1
        

    def join(self):
        _, _, t = self.config.weights()
        
        if self.t_wait == t:    
            self.t_wait = 0
            if self.on_site():
                self.state = 2
            else:
                self.state = 0
        else:
            self.pos += self.move
            self.t_wait += 1
    
    def still(self):
        self.freeze_movement()
    
    def leave(self):
        pass


    def change_position(self):
        
        self.there_is_no_escape()
        
        if self.state == 0:
            self.wandering()
        if self.state == 1:
            self.join()
        if self.state == 2:
            self.still()
        if self.state == 3:
            self.leave()
            
            
(
    Simulation(
        AggregationConfig(
            image_rotation=True,
            movement_speed=1.0,
            radius=15,
            seed=1,
            fps_limit =60,
        )
    )
        .spawn_site("site_medium.png", 375, 375)
        .batch_spawn_agents(30, Bee, images=["green.png"])
        .run()
)