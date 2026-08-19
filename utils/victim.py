from vmas.simulator.core import Landmark

class Victim(Landmark):

    def __init__(
        self,
        required_rescuers: int = 2,
        health: float = 100,
        decay_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.required_rescuers = required_rescuers
        self.health = health
        self.decay_rate = decay_rate
        self.rescued = False
    