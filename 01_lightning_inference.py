import pytorch_lightning as pl

import sys
sys.path.append("./")


if __name__ == "__main__":

    import os

    from code.dkt.args import parse_args
    from code.dkt.src.dataloader import Preprocess, get_loaders
    from dkt_lightning import DktLightning

    args = parse_args()

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    _, test_loader = get_loaders(args, None, test_data)

    # model = DktLightning(args)
    model = DktLightning.load_from_checkpoint(
        os.path.join(args.model_dir, args.model_name), args=args
    )
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=test_loader)
    print(predictions)