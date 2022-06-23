import pygame as pg
import vi
from vi import Agent, Simulation, Window, util
from vi.config import Config, dataclass, deserialize
import numpy as np
from numpy import random as r
import polars as pl
import seaborn as sns
from pygame.math import Vector2

@deserialize
@dataclass
class CompetitionConfig(Config):
    fox_energy: int = 500  # initial energy level of a fox
    hunger: int = 60  # time a fox has to wait before killing a rabbit
    r_rep: float = 0.8  # rabbit reproduction rate
    r_rep_buffer: int = 300  # time steps between each reproduction evaluation
    offspring: int = 2 # max offspring produced at reproduction
    hunting_radius: int = 15 # mating radius
    stress_deviation: float = 0.3

    def weights(self) -> tuple[int, int, float, int, int, int, float]:
        return (self.fox_energy, self.hunger, self.r_rep, self.r_rep_buffer, self.offspring, self.hunting_radius, self.stress_deviation)


class Fox(Agent):
    energy_t: int = 0
    hunger_t: int = 0
    alignment_max: int = 2

    config: CompetitionConfig


    def check_survival(self, energy):
        """
        Kills fox if energy is 0
        """
        if self.energy_t == energy:
            print(f"FOX: {self.id} died")
            self.kill()

    def consume(self, hunger, offspring, hunting_radius):
        """
        Checks whether there are nearby rabbits to consume and eventually does
        """
        if self.hunger_t == hunger:
            self.hunger_t = 0
            prey = (self.in_proximity_accuracy()
                   .filter_kind(Rabbit)
                   .first())

            if prey is not None:
                if prey[1] < hunting_radius:
                    prey[0].kill()
                    self.energy_t = 0
                    self.reproduction(offspring)
        
    def reproduction(self, offspring):
        """
        Reproduction of foxes gives some probability
        """
        mate = (self.in_proximity_accuracy()
               .without_distance()
               .filter_kind(Fox)
               .first())
        
        if mate is not None:
            for i in range(r.randint(1,offspring)):
                new_x = r.uniform()
                new_y = r.uniform()
                self.reproduce()
                self.move = Vector2(new_x, new_y)


    def change_position(self):
        self.there_is_no_escape()
        self.pos += self.move


    def update(self):
        self.save_data("agent_type", 1)
        
        energy, hunger, _, _, offspring, hunting_radius, _ = self.config.weights()  # init params

        self.check_survival(energy)  # kill fox if no energy is left
        self.consume(hunger, offspring, hunting_radius)

        self.energy_t += 1
        self.hunger_t += 1
        

class Rabbit(Agent):
    r_rep_buffer_t = 0
    
    config: CompetitionConfig

    def reproduction(self, r_rep, r_rep_buffer, offspring, stress_deviation):
        """
        Reproduction of Rabbits
        """
        if self.r_rep_buffer_t == r_rep_buffer:
            mate = (self.in_proximity_accuracy()
                    .without_distance()
                    .filter_kind(Rabbit)
                    .first())

            if mate is not None:
                roll = r.uniform()

                if self.in_proximity_accuracy().filter_kind(Fox).count() > 0:
                    r_bias = r.normal(0, stress_deviation)
                    r_rep -= abs(r_bias)



                if roll > r_rep:
                    self.r_rep_buffer_t = 0
                    for i in range(r.randint(1, offspring)):
                        new_x = r.uniform()
                        new_y = r.uniform()

                        self.reproduce().move = Vector2(new_x, new_y)

                    print(self.r_rep_buffer_t)
        else:
            self.r_rep_buffer_t += 1
        

    def change_position(self):

        self.there_is_no_escape()
        self.pos += self.move
        
        
    def update(self):
        self.save_data("agent_type", 2)
        
        _, _, r_rep, r_rep_buffer, offspring, _, stress_deviation = self.config.weights()  # init params
        self.reproduction(r_rep, r_rep_buffer, offspring, stress_deviation)
        


df = (
    Simulation(
        CompetitionConfig(
            image_rotation=True,
            movement_speed=1.5,
            radius=50,
            seed=30,
            fps_limit=60,
            duration=10000,
            window=Window.square(700),
        )
    )
        .batch_spawn_agents(100, Fox, images=["Fox.png"])
        .batch_spawn_agents(100, Rabbit, images=["Rabbit.png"])
        .run()
        .snapshots.groupby(["frame","agent_type"])
        .agg(pl.count("id").alias("agent"))
        .sort(["frame", "agent_type"])

)

plot = sns.relplot(x=df["frame"], y=df['agent'], hue=df["agent_type"], kind='line')

plot.savefig("plot.png", dpi=300)