import numpy as np
import torch
import torchvision

class InferenceHelper:
    """Helper class to simplify prediction through models.
    """
    def __init__(self, model_depth, model_seg, min_depth, max_depth, device):
        self.normalize = torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
        self.device = device
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.model_depth = model_depth.to(self.device)
        self.model_depth.eval()

        self.model_seg = model_seg.to(self.device)
        self.model_seg.eval()

    @torch.no_grad()
    def predict(self, img, is_depth = True, is_seg = True):
        """Predicts depth values for the provided image.
        Processing steps include:
            ¤ Normalizing the image.
            ¤ Predicting the depth on the flipped image and taking the average.

        Args:
            img: input RGB image with WxHxC format.

        Returns:
            bin_centers: predicted bins
            pred: depth predictions
        """
        pred_depth = None
        pred_seg = None
        bin_centers = None

        img = np.asarray(img) / 255.

        img = torch.from_numpy(img.transpose((2, 0, 1))) 
        img = self.normalize(img)
        img = img.unsqueeze(0).float().to(self.device)

        if is_depth:
            bin_centers, pred_depth = self._predict_depth(img)

        if is_seg:
            pred_seg = self._predict_seg(img)

        return bin_centers, pred_depth, pred_seg

    @torch.no_grad()
    def _predict_seg(self, image):
        pred = self.model_seg(image)
        pred = pred['out'].squeeze()
        pred = pred.argmax(0).detach().cpu()

        return pred

    @torch.no_grad()
    def _predict_depth(self, image):
        bins, pred = self.model_depth(image)
        pred = np.clip(
            pred.detach().cpu().numpy(),
            self.min_depth,
            self.max_depth
        )

        # flip and predict
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model_depth(image)[-1]
        pred_lr = np.clip(
            pred_lr.detach().cpu().numpy()[..., ::-1],
            self.min_depth,
            self.max_depth
        )
        
        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = torch.nn.functional.interpolate(
            torch.Tensor(final), image.shape[-2:],
            mode = 'bilinear', 
            align_corners = True
        ).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.detach().cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final
