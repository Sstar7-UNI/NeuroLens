from neurolens.TumorClass import TumorClass


class ClassMetrics:
    def __init__(self, label:TumorClass, precision:float, sensitivity:float,specificity:float, f1:float):
        self._label = label
        self._precision = precision
        self._sensitivity = sensitivity
        self._specificy = specificity
        self._f1 = f1
