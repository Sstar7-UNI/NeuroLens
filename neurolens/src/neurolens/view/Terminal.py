import glob
import os
import sys

from neurolens.controller.AugmentationConfig import AugmentationConfig
from neurolens.controller.MainController import MainController
from neurolens.controller.TrainingConfig import TrainingConfig

USAGE = """usage:
  neurolens learn        [augment=off|N] [preprocessing=none|learn|a,b,c] [batch_size=N|all] [model=NAME]
  neurolens evaluate     [model=NAME]
  neurolens runtime-test [model=NAME] src=PATH"""

class Terminal:
    def __init__(self,controller:MainController):
        self.controller = controller
    def run(self,arguments):
        if(len(arguments)>0):
            sys.exit(USAGE)
        options = self._parse(arguments[1:])
        try:
            if arguments[0] == "learn":
                self._learn(options)
            elif arguments[0] == "evaluate":
                self._evaluate(options)
            elif arguments[0] == "runtime-test":
                self._runtimeTest(options)
            else:
                sys.exit(USAGE)
        except ValueError as error:
            sys.exit("neurolens: " + str(error))

    def _learn(self,options):
        self._check(options, ["augment", "preprocessing", "batch_size", "model"])
        self._selectModel(options)
        copies = 0 #augmentare deactivata
        if(copies.get("augment","off") != "off"):
            copies = int(options["augment"])
        batch_size = 0
        if options.get("batch_size", "32") != "all":
            batchSize = int(options.get("batch_size", "32"))

        config = TrainingConfig(
            epochs=40, batchSize=batchSize, learningRate=0.001, weightDecay=0.0001,
            dropoutRate=0.0, decoupledDecay=False, patience=5,
            useAugmentation=copies > 0,
            augmentation=AugmentationConfig(imagesPerClass=100,
                                            copiesPerImage=max(copies, 1), seed=42),
            useClassWeights=False, seed=42)

        text = options.get("preprocessing", "none")
        if text == "learn":
            pipeline = self._controller.learnPreprocessing(config)
            print("learned preprocessing:", pipeline)
        elif text == "none":
            pipeline = []
        else:
            pipeline = text.split(",")
        self._controller.selectPreprocessing(pipeline)

        report = self._controller.train(config)
        print(f"best epoch {report.bestEpoch + 1} of {report.epochsCompleted}, "
              f"val macro F1 {report.bestValidationMacroF1:.4f}")
    def _evaluate(self,options):
        self._check(options, ["model"])
        self._selectModel(options)
        print(self._controller.evaluate().generateReport())
    def _runtimeTest(self,options):
        self._check(options, ["model", "src"])
        if "src" not in options:
            sys.exit("neurolens: runtime-test needs src=PATH")
        self._selectModel(options)

        source = options["src"]
        if os.path.isdir(source):
            images = sorted(glob.glob(os.path.join(source, "*.jpg")))
        else:
            images = [source]

        for path in images:
            if not self._controller.validateInput(path):
                print(f"{path}: not a valid image")
                continue
            result = self._controller.classify(path)
            print(f"{path} : {result.getPredictedLabel()}  "
                  f"{result.getConfidence():.1%} {result.inferenceTimeMs:.0f}  ms")

        def _selectModel(self, options):
            name = options.get("model", "best")
            if "/" not in name:
                name = "models/" + name
            self._controller.selectModel(name + ".weights.h5", name + ".json")

        def _parse(self, tokens):
            options = {}
            for token in tokens:
                parts = token.split("=", 1)
                if len(parts) != 2:
                    sys.exit(f"neurolens: expected key=value, got '{token}'")
                options[parts[0].lower()] = parts[1]
            return options

        def _check(self, options, allowed):
            for key in options:
                if key not in allowed:
                    sys.exit(f"neurolens: unknown option '{key}'\n{USAGE}")

def main():
        Terminal(MainController()).run(sys.argv[1:])

if __name__ == "__main__":
    main()
