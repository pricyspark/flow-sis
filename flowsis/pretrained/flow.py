import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from torch import Tensor
from numpy.typing import NDArray

def get_flow(
    model: ptlflow.BaseModel, 
    image1: NDArray,
    image2: NDArray, 
    io_adapter: IOAdapter | None = None
) -> tuple[Tensor, IOAdapter]:
    model.eval()
    
    images = [image1, image2]
    if io_adapter is None:
        io_adapter = IOAdapter(model, images[0].shape[:2])
        
    inputs = io_adapter.prepare_inputs(images)
    predictions = model(inputs)
    flows = predictions['flows'] # (1,1,2,H,W)
        
    return flows, io_adapter
