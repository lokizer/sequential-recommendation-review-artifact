from model.bsarec import BSARecModel
from model.caser import CaserModel
from model.gru4rec import GRU4RecModel
from model.sasrec import SASRecModel
from model.bert4rec import BERT4RecModel
from model.fmlprec import FMLPRecModel
from model.duorec import DuoRecModel
from model.fearec import FEARecModel
from model.sasfreqrec_v4_1 import SASFreqRecV41Model
from model.isolated_ablation import IsolatedAblationModel
from model.strict_specialized_ablation import StrictOriginalAblationModel

MODEL_DICT = {
    "bsarec": BSARecModel,
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "duorec": DuoRecModel,
    "fearec": FEARecModel,
    "duospectra_original": SASFreqRecV41Model,
    "duospectra_validity": IsolatedAblationModel,
    "duospectra_ablation": StrictOriginalAblationModel,
}
