from neurolens.TumorClass import TumorClass


class ClassMetrics:
    def __init__(self, label:TumorClass, precision:float, sensivity:float,specificy:float, f1:float):
        self._label = label
        self._precision = precision
        self._sensitivity = sensivity
        self._specificy = specificy
        self._f1 = f1
