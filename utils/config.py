from dataclasses import dataclass

@dataclass
class TrainConfig:
    device: str
    dataset: str
    nepoch: int
    K_layer: int
    nbatch: int = 16
    tau: float = 0.99
    sigma: float = 0.99
    init_param: str = None
    learning_rate: float = 1e-4
    nb_filter: int = 20
    R_init_type: str = 'ones'
    save: bool = False
    save_path: str = None
    norm_type: str = 'std' 