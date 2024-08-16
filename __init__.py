from .nodes_legacy import MiniCPM_VQA
from .nodes_polished import MiniCPM_VQA_Polished
from .image_nodes import MultipleImagesInput
from .util_nodes import LoadVideo,PreViewVideo
from .display_text_nodes import DisplayText
WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadVideo": LoadVideo,
    "PreViewVideo": PreViewVideo,
    "MultipleImagesInput": MultipleImagesInput,
    "MiniCPM_VQA": MiniCPM_VQA,
    "MiniCPM_VQA_Polished": MiniCPM_VQA_Polished,
    "DisplayText": DisplayText,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideo": "Load Video",
    "PreViewVideo": "PreView Video",
    "MultipleImagesInput": "Multiple Images Input",
    "MiniCPM_VQA": "MiniCPM VQA",
    "MiniCPM_VQA_Polished": "MiniCPM VQA Polished",
    "DisplayText": "Display Text",
}