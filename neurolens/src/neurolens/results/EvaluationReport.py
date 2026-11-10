
class EvaluationReport():
    def __init__(self,accuracy:float, empiricalError:float,macroPrecision:float,
                 macroSensivity:float, macroF1: float):
        self._accuracy = accuracy
        self._empiricalError = empiricalError
        self._macroPrecision = macroPrecision
        self._macroSensivity = macroSensivity
        self._macroF1 = macroF1

    def generateReport(self) -> str:
        return (f"Acuracy: {self._accuracy} \n Empirical error: {self._empiricalError} "
                f"\n Macro Precision : {self._macroPrecision} \n Macro sensivity: {self._macroSensivity} \n"
                f"Macro F1 : {self._macroF1} ")
