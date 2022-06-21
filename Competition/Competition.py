from enum import Enum, auto
import pygame as pg
import vi
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize
import numpy as np
from numpy import random as r
import polars as pl
import seaborn as sns



@deserialize
@dataclass
class AggregationConfig(Config):
    fox_energy: int = 200


    def weights(self) -> tuple[int]:
        return (self.a)


class Fox(Agent):
    energy_t = 0 

    config: AggregationConfig

    def wandering(self):
        pass

    def change_position(self):
        in_proximity = self.in_proximity_accuracy().count()
        energy = self.config.weights()
        self.there_is_no_escape()
        
        self.wandering()
        
class Rabbit(Agent):

    config: AggregationConfig

    def wandering(self):
        pass

    def change_position(self):
        in_proximity = self.in_proximity_accuracy().count()
        self.there_is_no_escape()
        
        self.wandering()

(
    AggregationConfig(
        image_rotation=True,
        movement_speed=2.0,
        radius=25,
        seed=30,
        fps_limit=60,
        duration=8000
    )
        .batch_spawn_agents(100, Fox, images=["Fox.png"])
        .batch_spawn_agents(100, Rabbit, images=["Rabbit.png"])
        .run()
        .snapshots.groupby(["frame","site_id"])
        .agg(pl.count("id").alias("agent"))
        .sort(["frame", "site_id"])

)

