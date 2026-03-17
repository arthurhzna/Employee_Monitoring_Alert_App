import torch
from typing import Optional

def get_tensor_value(tensor: Optional[torch.Tensor]) -> Optional[int]:
    if tensor is None:
        return None
    
    if tensor.is_cuda:  
        return int(tensor.cpu().item())
    else:  
        return int(tensor.item())