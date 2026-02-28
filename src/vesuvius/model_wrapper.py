import torch
from monai.networks.nets import SegResNet
from pathlib import Path

class ModelWrapper:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        self.model = None

    def build_segresnet(self):
        model = SegResNet(
            spatial_dims=3,
            in_channels=self.cfg.IN_CHANNELS,
            out_channels=self.cfg.OUT_CHANNELS,
            init_filters=self.cfg.INIT_FILTERS,
            blocks_down=[2, 3, 4, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        )
        return model.to(self.device)

    def load(self, checkpoint_path: str, model_builder=None):
        if model_builder is None:
            self.model = self.build_segresnet()
        else:
            self.model = model_builder().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        # Handle keys if saved with "module." prefix
        try:
            self.model.load_state_dict(state)
        except RuntimeError:
            # try to remove module prefix
            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            self.model.load_state_dict(new_state)
        self.model.eval()
        if self.logger:
            self.logger.info(f"Loaded model from {checkpoint_path}")
        return self.model

    def predict_sliding_window(self, volume_tensor, roi_size, sw_batch_size, overlap, mode='gaussian'):
        from monai.inferers import sliding_window_inference
        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=volume_tensor,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=overlap,
                mode=mode,
                device=self.device
            )
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        return preds
