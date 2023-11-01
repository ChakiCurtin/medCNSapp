from mmyolo.registry import VISUALIZERS
from mmyolo.registry import DATASETS
from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmdet.utils import register_all_modules
from mmengine.visualization import Visualizer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def registerstuff():
    register_all_modules()
    # @VISUALIZERS.register_module()
    # class DetLocalVisualizer(Visualizer):
    #     """MMDetection Local Visualizer.
    #     Args:
    #         name (str): Name of the instance. Defaults to 'visualizer'.
    #         image (np.ndarray, optional): the origin image to draw. The format
    #             should be RGB. Defaults to None.
    #         vis_backends (list, optional): Visual backend config list.
    #             Defaults to None.
    #         save_dir (str, optional): Save file dir for all storage backends.
    #             If it is None, the backend storage will not save any data.
    #         bbox_color (str, tuple(int), optional): Color of bbox lines.
    #             The tuple of color should be in BGR order. Defaults to None.
    #         text_color (str, tuple(int), optional): Color of texts.
    #             The tuple of color should be in BGR order.
    #             Defaults to (200, 200, 200).
    #         mask_color (str, tuple(int), optional): Color of masks.
    #             The tuple of color should be in BGR order.
    #             Defaults to None.
    #         line_width (int, float): The linewidth of lines.
    #             Defaults to 3.
    #         alpha (int, float): The transparency of bboxes or mask.
    #             Defaults to 0.8.
    #     Examples:
    #         >>> import numpy as np
    #         >>> import torch
    #         >>> from mmengine.structures import InstanceData
    #         >>> from mmdet.structures import DetDataSample
    #         >>> from mmdet.visualization import DetLocalVisualizer

    #         >>> det_local_visualizer = DetLocalVisualizer()
    #         >>> image = np.random.randint(0, 256,
    #         ...                     size=(10, 12, 3)).astype('uint8')
    #         >>> gt_instances = InstanceData()
    #         >>> gt_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
    #         >>> gt_instances.labels = torch.randint(0, 2, (1,))
    #         >>> gt_det_data_sample = DetDataSample()
    #         >>> gt_det_data_sample.gt_instances = gt_instances
    #         >>> det_local_visualizer.add_datasample('image', image,
    #         ...                         gt_det_data_sample)
    #         >>> det_local_visualizer.add_datasample(
    #         ...                       'image', image, gt_det_data_sample,
    #         ...                        out_file='out_file.jpg')
    #         >>> det_local_visualizer.add_datasample(
    #         ...                        'image', image, gt_det_data_sample,
    #         ...                         show=True)
    #         >>> pred_instances = InstanceData()
    #         >>> pred_instances.bboxes = torch.Tensor([[2, 4, 4, 8]])
    #         >>> pred_instances.labels = torch.randint(0, 2, (1,))
    #         >>> pred_det_data_sample = DetDataSample()
    #         >>> pred_det_data_sample.pred_instances = pred_instances
    #         >>> det_local_visualizer.add_datasample('image', image,
    #         ...                         gt_det_data_sample,
    #         ...                         pred_det_data_sample)
    #     """
    #     def __init__(self,
    #                 name: str = 'visualizer',
    #                 image: Optional[np.ndarray] = None,
    #                 vis_backends: Optional[Dict] = None,
    #                 save_dir: Optional[str] = None,
    #                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
    #                 text_color: Optional[Union[str,
    #                                             Tuple[int]]] = (200, 200, 200),
    #                 mask_color: Optional[Union[str, Tuple[int]]] = None,
    #                 line_width: Union[int, float] = 3,
    #                 alpha: float = 0.8) -> None:
    #         super().__init__(
    #             name=name,
    #             image=image,
    #             vis_backends=vis_backends,
    #             save_dir=save_dir)
    #         self.bbox_color = bbox_color
    #         self.text_color = text_color
    #         self.mask_color = mask_color
    #         self.line_width = line_width
    #         self.alpha = alpha
    #         # Set default value. When calling
    #         # `DetLocalVisualizer().dataset_meta=xxx`,
    #         # it will override the default value.
    #         self.dataset_meta = {}
    @DATASETS.register_module()
    class MoNuSegDataset(YOLOv5CocoDataset):
        """MoNuSeg dataset."""
        METAINFO = {
            'classes': ("nucleus"),
            'palette':[
                    (0, 255, 0),
            ],  # This has been changed, lets see what it does
        }
        def __init__(
                self,
                #img_suffix=".png",
                seg_map_suffix=".png",
                return_classes=False,
                **kwargs,
            ) -> None:
                self.return_classes = return_classes
                super().__init__(
                    #img_suffix=img_suffix,
                    seg_map_suffix=seg_map_suffix,
                    **kwargs,
                )