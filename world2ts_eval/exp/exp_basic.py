import torch

from models import CastFSRZero, Chronos2, DirectLLM, Naive, PatchTST


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "LastValue": Naive,
            "SeasonalNaive": Naive,
            "PatchTST": PatchTST,
            "DirectLLM": DirectLLM,
            "CastFSRZero": CastFSRZero,
            "Chronos2": Chronos2,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        return self.model_dict[self.args.model].Model(self.args).float()

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device(f"cuda:{self.args.gpu}")
        return torch.device("cpu")
