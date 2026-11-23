from neurolens.data_management.TumorClass import TumorClass


class ClassMetrics:
    def __init__(self, label:TumorClass, precision:float, sensitivity:float,specificity:float, f1:float):
        self._label = label
        self._precision = precision
        self._sensitivity = sensitivity
        self._specificy = specificity
        self._f1 = f1

    @property
    def precision(self):
        return self._precision

    @property
    def sensitivity(self):
        return self._sensitivity

    @property
    def f1(self):
        return self._f1