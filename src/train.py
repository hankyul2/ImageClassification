from pytorch_lightning import Trainer

from src.model.models import get_model
from src.dataset.base_data_module import get_cifar, convert_to_dataloader
from src.task.base_vision_system import BaseVisionSystem


def run(args):
    # step 1. prepare dataset
    train_ds, valid_ds, test_ds = get_cifar(args.dataset, size=args.img_size)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size,
                                              num_workers=args.num_workers, shuffle=False)

    # step 2. load model
    model = get_model(args.model_name, nclass=len(train_ds.dataset.classes), pretrained=args.pretrained, dropout=args.dropout)

    # step 3. load ml system (loss, optimizer, lr scheduler)
    task = BaseVisionSystem(model, batch_step=len(train_dl), max_epoch=args.max_epoch, lr=args.lr)

    # step 4. train
    trainer = Trainer(gpus=8, accelerator='ddp', max_epochs=args.max_epoch)
    trainer.fit(task, train_dl, valid_dl)

    # step 5. evaluate
    trainer.test(test_dataloaders=test_dl)