import torch

def reshape_image_by_patch(init_image):
    init_shape = init_image.shape[2:]
    final_shape = (224, 224)
    patch_size = 32
    final_image = torch.zeros(init_image.shape[:2] + final_shape)
    remaining_w = init_shape[1]
    idx = 0
    while remaining_w > 0:
        w_limit = min(remaining_w, final_shape[1])
        final_image[:, :, idx*init_shape[0]:(idx+1)*init_shape[0], 0:0+w_limit] = init_image[:, :, 0:0+init_shape[0], idx*final_shape[1]:idx*final_shape[1]+w_limit]
        remaining_w -= w_limit
        idx += 1

    return final_image
