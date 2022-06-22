import pygame as pg
import vi
from vi import Agent, Simulation, Window, util
from vi.config import Config, dataclass, deserialize
import numpy as np
from numpy import random as r
import polars as pl
import seaborn as sns


@deserialize
@dataclass
class CompetitionConfig(Config):
    fox_energy: int = 600  # initial energy level of a fox
    hunger: int = 30  # time a fox has to wait before killing a rabbit
    r_rep: float = 0.2  # rabbit reproduction rate
    f_rep: float = 0.2  # fox reproduction rate
    r_death: float = 0.8  # rabbit death rate
    f_death: float = 0.3  # fox death rate

    def weights(self) -> tuple[int, int, float, float, float, float]:
        return (self.fox_energy, self.hunger, self.r_rep, self.f_rep, self.r_death, self.f_death)


class Fox(Agent):
    energy_t: int = 0
    hunger_t: int = 0

    config: CompetitionConfig

    def check_survival(self, energy):
        """
        Kills fox if energy is 0
        """
        if self.energy_t == energy:
            print(f"FOX: {self.id} died")
            self.kill()

    def consume(self, agent, r_death):
        """
        Eat a rabbit given some probability
        might need some others calls whenever exploring beyond base case.
        for now it just kills the rabbits.
        """
        roll = r.uniform()
        if r_death > roll:
            print('KILL')
            self.energy_t = 0
            agent.kill()
        
    def reproduction(self, f_rep):
        """
        Reproduction of foxes gives some probability
        """
        sample = r.uniform()
        if sample > f_rep:
            self.reproduce()


    def change_position(self):
        self.pos += self.move
        
    def update(self):
        in_proximity = self.in_proximity_accuracy()
        count = self.in_proximity_accuracy().count()
        energy, hunger, _, f_rep, r_death, _ = self.config.weights()  # init params
        self.there_is_no_escape()

        self.check_survival(energy)  # kill fox if no energy is left
        if self.hunger_t == hunger:
            self.hunger_t = 0
        for agent, _ in in_proximity:  # check for rabbits within radius x
            if type(agent) == Rabbit and count == 1 and self.hunger_t == 0:
                self.consume(agent, r_death)
            if type(agent) == Fox and self.energy_t != energy:
                # self.reproduction(f_rep)
                pass

        self.energy_t += 1
        self.hunger_t += 1

class Rabbit(Agent):
    config: CompetitionConfig

    def wandering(self):
        """
        Random Walk
        """

        self.pos = self.pos + self.move
        return self.pos

    def reproduction(self, r_rep):
        """
        Reproduction of Rabbits
        """
        sample = r.uniform()
        if sample > r_rep:
            self.reproduce()

    def change_position(self):
        in_proximity = self.in_proximity_accuracy()
        count = self.in_proximity_accuracy().count()
        _, _, r_rep, _, _, _ = self.config.weights()  # init params

        for agent, _ in in_proximity: # reproduction of rabbits
            if count == 1 and type(agent) == Rabbit:
                pass
                #self.reproduction(r_rep)

        self.there_is_no_escape()
        self.wandering()


(
    Simulation(
        CompetitionConfig(
            image_rotation=True,
            movement_speed=1.0,
            radius=25,
            seed=30,
            fps_limit=60,
            window=Window.square(1000),
        )
    )
        .batch_spawn_agents(100, Fox, images=["red.png"])
        .batch_spawn_agents(100, Rabbit, images=["green.png"])
        .run()
    # .snapshots.groupby(["frame","site_id"])
    # .agg(pl.count("id").alias("agent"))
    # .sort(["frame", "site_id"])

)
