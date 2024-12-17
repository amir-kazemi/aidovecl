from ultralytics import YOLO

patience = 20
batch = 32
lr0 = 0.01

def train_real():
    model = YOLO('yolov8n.pt')
    results = model.train(data='./datasets/real_seeded_split/real_seeded_split.yaml',
                          epochs=1000,
                          imgsz=512,
                          seed = 0,
                          mosaic = 0,
                          mixup = 0,
                          patience = patience,
                          batch = batch,
                          lr0=lr0,
                          name='train_real')

def train_real_mixup():
    model = YOLO('yolov8n.pt')
    results = model.train(data='./datasets/real_seeded_split/real_seeded_split.yaml',
                          epochs=1000,
                          imgsz=512,
                          seed = 0,
                          mosaic = 0,
                          mixup = 1,
                          patience = patience,
                          batch = batch,
                          lr0=lr0,
                          name='train_real_mixup')

def train_real_mosaic():
    model = YOLO('yolov8n.pt')
    results = model.train(data='./datasets/real_seeded_split/real_seeded_split.yaml',
                          epochs=1000,
                          imgsz=512,
                          seed = 0,
                          mosaic = 1,
                          mixup = 0,
                          patience = patience,
                          batch = batch,
                          lr0=lr0,
                          name='train_real_mosaic')

def train_augmented():
    model = YOLO('yolov8n.pt')
    results = model.train(data='./datasets/augmented/augmented.yaml',
                          epochs=1000,
                          imgsz=512,
                          seed = 0,
                          mosaic = 0,
                          mixup = 0,
                          patience = patience,
                          batch = batch*2,
                          lr0=lr0*2,
                          name = 'train_augmented')

def train_augmented_mixup():
    model = YOLO('yolov8n.pt')
    results = model.train(data='./datasets/augmented/augmented.yaml',
                          epochs=1000,
                          imgsz=512,
                          seed = 0,
                          mosaic = 0,
                          mixup = 1,
                          patience = patience,
                          batch = batch*2,
                          lr0=lr0*2,
                          name='train_augmented_mixup')

def train_augmented_mosaic():
    model = YOLO('yolov8n.pt')
    results = model.train(data='./datasets/augmented/augmented.yaml',
                          epochs=1000,
                          imgsz=512,
                          seed = 0,
                          mosaic = 1,
                          mixup = 0,
                          patience = patience,
                          batch = batch*2,
                          lr0=lr0*2,
                          name='train_augmented_mosaic')

def test_real():
    best_model = YOLO("./runs/detect/train_real/weights/best.pt")
    results = best_model.val(
        data= './datasets/real_seeded_split/real_seeded_split.yaml',
        split = 'test',
        imgsz=512,
        name='test_real',
        save_json=True,
        half = True
    )
    return results

def test_real_mixup():
    best_model = YOLO("./runs/detect/train_real_mixup/weights/best.pt")
    results = best_model.val(
        data= './datasets/real_seeded_split/real_seeded_split.yaml',
        split = 'test',
        imgsz=512,
        name='test_real_mixup',
        save_json=True,
        half = True
    )
    return results


def test_real_mosaic():
    best_model = YOLO("./runs/detect/train_real_mosaic/weights/best.pt")
    results = best_model.val(
        data= './datasets/real_seeded_split/real_seeded_split.yaml',
        split = 'test',
        imgsz=512,
        name='test_real_mosaic',
        save_json=True,
        half = True
    )
    return results


def test_augmented():
    best_model = YOLO("./runs/detect/train_augmented/weights/best.pt")
    results = best_model.val(
        data= './datasets/real_seeded_split/real_seeded_split.yaml',
        split = 'test',
        imgsz=512,
        name='test_augmented',
        save_json=True,
        half = True
    )
    return results

def test_augmented_mixup():
    best_model = YOLO("./runs/detect/train_augmented_mixup/weights/best.pt")
    results = best_model.val(
        data= './datasets/real_seeded_split/real_seeded_split.yaml',
        split = 'test',
        imgsz=512,
        name='test_augmented_mixup',
        save_json=True,
        half = True
    )
    return results

def test_augmented_mosaic():
    best_model = YOLO("./runs/detect/train_augmented_mosaic/weights/best.pt")
    results = best_model.val(
        data= './datasets/real_seeded_split/real_seeded_split.yaml',
        split = 'test',
        imgsz=512,
        name='test_augmented_mosaic',
        save_json=True,
        half = True
    )
    return results




