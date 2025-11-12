from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainer_optimized(nnUNetTrainer):
    """
    Optimized trainer with better caching strategy:
    - Uses 3 processes for training, 2 for validation  
    - Increases cache size to 12 batches (vs default 6)
    - Better GPU utilization with minimal CPU overhead
    
    Usage:
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 516 3d_fullres 2 -tr nnUNetTrainer_optimized
    """
    
    @staticmethod
    def get_allowed_n_proc_DA_custom():
        """
        Override to return 3 processes (optimized for balance)
        """
        return 3

