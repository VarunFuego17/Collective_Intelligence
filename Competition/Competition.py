from vi import Agent, Simulation, Window, util, HeadlessSimulation
from vi.config import Config, dataclass, deserialize
from numpy import random as r
import polars as pl
import seaborn as sns
from pygame.math import Vector2
import matplotlib.pyplot as plt

@deserialize
@dataclass
class CompetitionConfig(Config):
    fox_energy: int = 500  # initial energy level of a fox
    hunger: int = 60  # time a fox has to wait before killing a rabbit
    r_rep: float = 0.8  # rabbit reproduction rate
    r_rep_buffer: int = 400  # time steps between each reproduction evaluation
    offspring: int = 2 # max offspring produced at reproduction
    reach_radius: int = 15 # mating and hunting radius
    stress_deviation: float = 0.3

    def weights(self) -> tuple[int, int, float, int, int, int, float]:
        return (self.fox_energy, self.hunger, self.r_rep, self.r_rep_buffer, self.offspring, self.reach_radius, self.stress_deviation)


class Fox(Agent):
    energy_t: int = 0
    hunger_t: int = 0

    config: CompetitionConfig


    def check_survival(self, energy):
        """
        Kills fox if energy is 0
        """
        if self.energy_t == energy:
            self.kill()

    def consume(self, hunger, offspring, reach_radius):
        """
        Checks whether there are nearby rabbits to consume and eventually does
        """
        if self.hunger_t == hunger:
            self.hunger_t = 0
            prey = (self.in_proximity_accuracy()
                   .filter_kind(Rabbit)
                   .first())

            if prey is not None and prey[1] < reach_radius:
                prey[0].kill()
                self.energy_t = 0
                self.reproduction(offspring, reach_radius)
        
    def reproduction(self, offspring, reach_radius):
        """
        Reproduction of foxes gives some probability
        """
        mate = (self.in_proximity_accuracy()
               .filter_kind(Fox)
               .first())
        
        if mate is not None and mate[1] < reach_radius:
            for i in range(r.randint(1,offspring)):
                new_x = r.uniform(-1.5, 1.5)
                new_y = r.uniform(-1.5, 1.5)
                self.reproduce()
                self.move = Vector2(new_x, new_y)


    def change_position(self):
        self.there_is_no_escape()
        self.pos += self.move


    def update(self):
        self.save_data("agent_type", 1)
        
        energy, hunger, _, _, offspring, reach_radius, _ = self.config.weights()  # init params

        self.check_survival(energy)  # kill fox if no energy is left
        self.consume(hunger, offspring, reach_radius)

        self.energy_t += 1
        self.hunger_t += 1


class Rabbit(Agent):
    r_rep_buffer_t = 0
    
    config: CompetitionConfig

    def reproduction(self, r_rep, r_rep_buffer, offspring, stress_deviation, reach_radius):
        """
        Reproduction of Rabbits
        """
        if self.r_rep_buffer_t == r_rep_buffer:
            mate = (self.in_proximity_accuracy()
                    .filter_kind(Rabbit)
                    .first())

            if mate is not None and mate[1] < reach_radius:
                roll = r.uniform()

                if self.in_proximity_accuracy().filter_kind(Fox).count() > 0:
                    r_bias = r.normal(0, stress_deviation)
                    r_rep -= abs(r_bias)

                if roll > r_rep:
                    self.r_rep_buffer_t = 0
                    for i in range(r.randint(1, offspring)):
                        new_x = r.uniform(-1.5,1.5)
                        new_y = r.uniform(-1.5,1.5)

                        self.reproduce().move = Vector2(new_x, new_y)
        else:
            self.r_rep_buffer_t += 1
        

    def change_position(self):

        self.there_is_no_escape()
        self.pos += self.move
        
        
    def update(self):
        self.save_data("agent_type", 2)
        
        _, _, r_rep, r_rep_buffer, offspring, reach_radius, stress_deviation = self.config.weights()  # init params
        self.reproduction(r_rep, r_rep_buffer, offspring, stress_deviation, reach_radius)
        
energy_values = [500, 600, 700]
hunger_values = [40, 60, 80]
#reach_values = [10, 15]
stress_dev_values = [0.5, 0.6]

for energy_value in energy_values:
    for hunger_value in hunger_values:
        for stress_dev_value in stress_dev_values:

            df = (
                HeadlessSimulation(
                    CompetitionConfig(
                        image_rotation=True,
                        movement_speed=1.5,
                        radius=45,
                        seed=30,
                        fps_limit=60,
                        duration=8000,
                        window=Window.square(700),
                        fox_energy=energy_value,
                        hunger=hunger_value,
                        stress_deviation=stress_dev_value,

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

            plot.savefig(f"energy{energy_value}_hunger{hunger_value}_stress{stress_dev_value}.png", dpi=300)

            plt.close()