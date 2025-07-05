import os
import random
import gc
import typing as ty

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import cv2

from clip_interactively.segm import CLIPForSegmentation


def get_image_paths(dir_path, exts=('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):
    image_paths = []
    for fname in os.listdir(dir_path):
        if fname.lower().endswith(exts):
            image_paths.append(os.path.join(dir_path, fname))
    return image_paths


def get_color_map(classes_names: ty.List[str]) -> ty.Dict[str, ty.Tuple[int, int, int]]:

    n_classes = len(classes_names)
    rng = np.random.RandomState(42)

    map_cls_ind_to_color = {
        #0: (0, 0, 0) # background
    }
    #for idx in range(1, n_classes+1):
    for idx in range(n_classes):
        if idx == 0:
            map_cls_ind_to_color[idx] = (0, 0, 0)
        else:
            map_cls_ind_to_color[idx] = tuple(int(x) for x in rng.randint(0, 256, size=3))
    
    return map_cls_ind_to_color


def save_classes_legend(
        classes_names: ty.List[str],
        map_cls_ind_to_color: ty.Dict[str, ty.Tuple[int, int, int]], 
        save_path: str
    ):

    classes_names = {ind:cls for ind, cls in enumerate(classes_names)}

    rect_w, rect_h = 40, 40
    padding = 10
    font_size = 24

    # Вычисляем размер картинки
    width = rect_w + 3 * padding + max(len(label) for label in classes_names.values()) * (font_size // 2)
    height = len(classes_names) * (rect_h + padding) + padding

    # Создаём картинку
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    # Рисуем легенду
    for i, (cls_idx, label) in enumerate(classes_names.items()):
        y = padding + i * (rect_h + padding)
        # Цветной прямоугольник
        draw.rectangle([padding, y, padding + rect_w, y + rect_h], fill=map_cls_ind_to_color[cls_idx])
        # Подпись
        draw.text((padding * 2 + rect_w, y + (rect_h - font_size) // 2), label, fill="black", font=font)

    # Сохраняем и показываем
    img.save(f"{save_path}/__class_legend__.png")


def save_colored_mask(
        mask_tensor: torch.Tensor,
        class_names: list[str],
        save_path: str,
        colormap: dict[int, tuple[int,int,int]] | None = None
    ):
    """
    Args:
        mask_tensor: torch.Tensor of shape (1, H, W) or (H, W), with integer labels:
            0 = background, 1..N = classes in class_names.
        class_names: list of class names in order [‘cat’, ‘bird’, …].
        save_path: where to save the PNG (e.g. 'mask_colored.png').
        colormap: optional dict mapping label→RGB; if None, a default is generated.
    """
    # squeeze to (H, W)
    if mask_tensor.dim() == 3:
        mask = mask_tensor.squeeze(0)
    else:
        mask = mask_tensor
    mask_np = mask.detach().cpu().numpy()
    
    # Create RGB array
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, color in colormap.items():
        color_mask[mask_np == lbl] = color
    
    # Save
    res = Image.fromarray(color_mask)
    res = res.transpose(Image.ROTATE_270)
    res.save(save_path)
    print(f"Saved colored mask to {save_path}")


def overlay_mask_on_image(
    img: Image.Image,
    mask_tensor: torch.Tensor,
    class_names: list[str],
    colormap: dict[int, tuple[int, int, int]],
    save_path: str,
    alpha: float = 0.5
):
    """
    Overlays a semi-transparent color mask on the original image.
    Mask is only applied to non-background pixels (class 0 is background).
    """
    # 1) Squeeze out any batch/channel dims
    if mask_tensor.dim() == 3:
        mask = mask_tensor.squeeze(0)
    else:
        mask = mask_tensor

    # 2) Bring mask to CPU numpy (H_mask x W_mask)
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)

    # 3) Convert PIL image to numpy (H_img x W_img x 3)
    img_np = np.array(img.convert("RGB"), copy=True)
    h_img, w_img = img_np.shape[:2]

    # 4) If mask and image shapes differ, resize mask with nearest neighbor
    if mask_np.shape != (h_img, w_img):
        mask_img = Image.fromarray(mask_np, mode="L")
        mask_img = mask_img.resize((w_img, h_img), resample=Image.NEAREST)
        mask_np = np.array(mask_img)

    # 5) Prepare output and overlay
    out = img_np.copy()
    for lbl, color in colormap.items():
        if lbl == 0:
            continue  # skip background
        mask_area = (mask_np == lbl)
        # Blend: α·color + (1−α)·original
        blended = (
            alpha * np.array(color)[None, None, :]
            + (1 - alpha) * img_np[mask_area]
        ).astype(np.uint8)
        out[mask_area] = blended
    #out = out.transpose((1, 0, 2))

    # 6) Save result
    res = Image.fromarray(out)
    res = res.transpose(Image.ROTATE_270)
    res.save(save_path)
    print(f"Saved overlay to {save_path}")


if __name__ == '__main__':

    input_dir_path = 'tmp5'

    save_path = 'tmp5_segm_res'
    os.makedirs(save_path, exist_ok=True)

    # class_names = [
    #     'background',
    #     'cat',
    #     'bird',
    #     'person',
    #     'antelope',
    #     'tree',
    #     'bush',
    #     'cheetah',
    #     'elephant',
    #     'eat',
    #     'sky',
    #     'anime'
    # ]

    class_names = [
        'background',
        'cat',
        'waterfall',
        'water',
        'conifers',
        'car',
        'human',
        'road sign',
        'tree',
        'bush',
        'sky',
        'house with unusual (ex. blue) color',
        'house with ordinary color',
        'bridge',
        'dish',
        'plate',
        'bread',
        'fruit or berries',
        'grass',
        'ground',
        'cactus'
    ]
    size = (896, 2048)

    #class_names_with_backgr = ['background'] + class_names
    map_cls_ind_to_color = get_color_map(class_names)
    with open(f'{save_path}/class_names.txt', 'w') as f:
        class_names_str = [f"{ind} <===> {item}" for ind, item in enumerate(class_names)]
        f.write('\n'.join(class_names_str))
        f.write('\n')
        f.write({ind: color for ind, color in map_cls_ind_to_color.items()}.__str__())
    save_classes_legend(class_names, map_cls_ind_to_color, save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPForSegmentation(
        class_names=class_names,
        size=size
    )

    for image_path in tqdm(get_image_paths(input_dir_path)):
        base_name = os.path.basename(image_path)
        name_wo_ext = os.path.splitext(base_name)[0]
    
        raw_image = Image.open(image_path)
        print(f"Input image shape: {raw_image.size}")
        try:
            prep_image = model.data_preprocessor(raw_image)
        except IndexError:
            continue
        tensor = torch.unsqueeze(prep_image, dim=0).to(device)

        print(f"Input tensor shape: {tensor.shape}")
        seg_preds = model.predict(inputs=tensor)
        print(f"Output tensor shape: {len(seg_preds)}")
        seg_pred = seg_preds[0]
        print(f"Output tensor shape: {seg_pred.shape}")

        uniques = torch.unique(seg_pred)
        print(f"Uniques: {uniques}")

        save_colored_mask(
            mask_tensor=seg_pred, 
            save_path=f'{save_path}/{name_wo_ext}_mask.png', 
            class_names=class_names,
            colormap=map_cls_ind_to_color
        )
        
        # class_ind = class_inds[0]
        # print(f"Class ind: {class_ind}")

        overlay_mask_on_image(
            img=raw_image, 
            mask_tensor=seg_pred,
            class_names=class_names,
            colormap=map_cls_ind_to_color,
            save_path=f'{save_path}/{name_wo_ext}_overlay.png',
            alpha=0.5
        )

        del raw_image, prep_image, tensor, seg_pred
        gc.collect()
        torch.cuda.empty_cache()