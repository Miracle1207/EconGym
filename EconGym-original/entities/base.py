from abc import ABC, abstractmethod


class BaseEntity(ABC):
    name=""
    type=None


    def __init__(self):
        assert self.name


    def reset(self):
        pass

  

    def get_metrics(self):
        pass




