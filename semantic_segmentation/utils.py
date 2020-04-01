from dataloader import valid_labels, CityScapesDataset
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import DataLoader

class_weights = np.array([1.12990371e-04, 4.58409167e-02, 1.30404310e-02, 1.50842667e-02,
                          1.34239003e-02, 2.85600678e-03, 1.21220385e-02, 3.26399687e-01,
                          5.38691021e-02, 6.26141364e-03, 1.80143693e-03, 2.02056519e-01,
                          5.80210614e-03, 7.76630145e-03, 8.77063014e-05, 2.86265413e-03,
                          5.38998291e-04, 1.08653968e-02, 8.01201828e-05, 1.83956371e-03,
                          4.88027893e-03, 1.41013007e-01, 1.02499210e-02, 3.55792079e-02,
                          1.07911733e-02, 1.19620604e-03, 6.19212377e-02, 2.36772938e-03,
                          2.08210184e-03, 3.99641630e-04, 2.08457979e-04, 2.06185269e-03,
                          8.73397699e-04, 3.66423038e-03])

inverted_weights = np.log(1 / class_weights)


def iou_class(pred, target, cls):
    pred_l = pred == cls
    target_l = target == cls

    intersection = (pred_l * target_l).sum()
    union = (pred_l | target_l).sum()

    if union == 0:
        return float('nan')  # if there is no ground truth, do not include in evaluation
    else:
        return (intersection * 1.0 / union).cpu()


def iou(pred, target):
    ious = []
    for cls in valid_labels:
        io = iou_class(pred, target, cls)
        if not np.isnan(io):
            ious.append(io)
    return np.average(ious)


def get_class_distribution():
    train_dataset = CityScapesDataset(csv_file='train0.csv')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              num_workers=4)

    distribution = np.zeros((34,), dtype='int64')
    for iter, (X, tar, Y) in enumerate(train_loader):
        print(iter)
        labels, counts = np.unique(Y, return_counts=True)
        distribution[labels] += counts

    print(distribution / distribution.sum())


def pixel_acc(pred, target):
    count = 0
    total = 0
    for label in valid_labels:
        pred_l = pred == label
        target_l = target == label

        count += (pred_l * target_l).sum()
        total += target_l.sum()

    return (count * 1.0 / total).cpu()


def overlayImages(foreground, background):
    background = Image.open(background)
    overlay = Image.open(foreground)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.5)
    new_img.save("new.png", "PNG")
