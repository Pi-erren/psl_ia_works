from abc import ABC, abstractmethod


class Problem(ABC):

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def step(self):
        pass
