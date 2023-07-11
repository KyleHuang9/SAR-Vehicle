dataset = dict(
    txt_dir="/home/hyl/home/DL_Homework/SAR-Vehicle/DL_dataset",
    img_size=(32, 32),
    nc=3,
)

model = dict(
    model=None,
    base_channels=32,
)

params = dict(
    epoch=100,
    batch_size=32,
    lr0=0.002,
    lr1=0.0001,
    eval_interval=5,
    warm_up=3,
    momentum=0.937,
    weight_decay=0.005,
)

data_aug = dict(
    augment=1.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.5,
    gaussian=0.0,
    gau_d=20,
    degrees=30.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
)