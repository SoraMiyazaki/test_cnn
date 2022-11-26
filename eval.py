import torch
from torch.utils.data import DataLoader
from dataset.poc import CustomImageDataset
from torchvision.models.densenet import DenseNet
from loss.lossfunc import LossFunc

batch_size = 64
pred_path = "pred.csv"
test_path = "../data/poc/test.csv"
img_dir = "../data/poc/"
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_ds = CustomImageDataset(test_path, img_dir)
eval_dl = DataLoader(eval_ds, batch_size=batch_size)
model = DenseNet(num_classes=250).to(device)
statedict = torch.load("../data/statedict/statedict2000.pt")
model.load_state_dict(statedict)
lossfunc = LossFunc()
model.eval()

with torch.no_grad():
    with open(pred_path, mode="w") as f:
        for feature, truth in eval_dl:
            feature = feature.to(device)
            truth = truth.to(device)
            batch_pred = model(feature)

            pred: torch.Tensor
            for pred in batch_pred:
                f.write(",".join(map(str, pred)) + "\n")
