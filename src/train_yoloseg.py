from argparse import ArgumentParser

from ultralytics import YOLO
from loguru import logger

from src.consts import DATA_DIR
from src.training_args import get_training_args


def main(data_dir: str, yolo_ver: str):
    yolo_file = DATA_DIR / data_dir / f"{data_dir}.yaml"
    logger.info(f"Training YOLOv5 on `{yolo_file}` with `{yolo_ver}` version")

    training_args = get_training_args()

    model = YOLO(yolo_ver + ".pt")
    results = model.train(
        data=yolo_file,
        epochs=training_args.epochs,
        batch=training_args.batch_size,
        device=training_args.device,
        lr0=training_args.learning_rate,
        momentum=training_args.momentum,
        resume=training_args.resume,
        plots=training_args.plots,
        imgsz=256,
    )

    logger.info(results)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="seep_detection")
    argparser.add_argument("--yolo-ver", type=str, default="yolo11m-seg")

    args = argparser.parse_args()
    main(args.data_dir, args.yolo_ver)
