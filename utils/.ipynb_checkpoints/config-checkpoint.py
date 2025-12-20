from dataclasses import dataclass

@dataclass
class TrainConfig:
    device: str
    dataset_train: str
    dataset_validation: str
    choix_model: str
    choix_loss: str
    nepoch: int
    K_layer: int
    nbatch_train: int = 16
    nbatch_validation: int = 4
    lmb_init: float = 1
    tau: float = 0.99
    sigma: float = 0.99
    init_param: str = None
    scheduler: bool = False
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    largeur: int = 20
    start_end_oplin_size: int = 20
    batch_norm: bool = False
    norm_type: str = 'max'
    R_init_type: str = 'MLE'
