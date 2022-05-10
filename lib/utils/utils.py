import torch

def process_model_dict(weights_path):
    """Loads and processes the pre-trained model checkpoint dictionary to maintain compatibility to
     older naming of architecture blocks

    Args:
        weights_path: path to pre-trained weights

    Returns:
        modified: model checkpoints
    """
    ckpt = torch.load(weights_path, map_location='cpu')

    if 'model' in ckpt:
        ckpt = ckpt['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                            'adaptive_bins_layer.conv3x3.')
            modified[k_] = v

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):
            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                            'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v

        else:
            modified[k] = v  # else keep the original

    return modified
