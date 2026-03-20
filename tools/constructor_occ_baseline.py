import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.networks.resnet_networks.network import _resnet as build_bb_resnet
from models.networks.resnet_networks.network import BasicBlock as BBBasicBlock
from models.networks.resnet_topo_networks.network import _resnet as build_topo_resnet
from models.networks.resnet_topo_networks.network import BasicBlock as TopoBasicBlock


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone constructor baseline with optional occ-from-sdf input")
    parser.add_argument("--task", choices=["topo", "bb"], required=True)
    parser.add_argument("--input_variant", choices=["sdf", "sdf_occ"], default="sdf_occ")
    parser.add_argument("--dataroot", type=str, default="./data")
    parser.add_argument("--dataset_mode", type=str, default="mof_250k")
    parser.add_argument("--config", type=str, default="./configs/mof_constructor.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--trunc_thres", type=float, default=0.2)
    parser.add_argument("--max_dataset_size", type=int, default=2147483648)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--occ_source_channel", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def split_mof(mof_name: str):
    tokens = mof_name.split("+")
    if len(tokens) == 3:
        return tokens[0], tokens[1], "N0", tokens[2]
    return tokens[0], tokens[1], tokens[2], tokens[3]


def mof_to_id(encoders, mof_name: str):
    topo, node1, node2, edge = split_mof(mof_name)
    topo_id = encoders["topo"].transform([topo])
    node1_id = encoders["node"].transform([node1])
    node2_id = encoders["node"].transform([node2])
    edge_id = encoders["edge"].transform([edge])
    return np.hstack((topo_id, node1_id, node2_id, edge_id))


def resolve_paths(dataroot: Path):
    sdf_candidates = [
        dataroot / "sdfs" / "resolution_32",
        dataroot / "resolution_32",
    ]
    lcd_candidates = [dataroot / "lcd_data.txt", dataroot / "properties" / "lcd_data.txt"]
    pld_candidates = [dataroot / "pld_data.txt", dataroot / "properties" / "pld_data.txt"]
    vf_candidates = [dataroot / "vf_data.txt", dataroot / "properties" / "vf_data.txt"]

    sdf_dir = next((path for path in sdf_candidates if path.exists()), None)
    if sdf_dir is None:
        raise FileNotFoundError(f"Could not find resolution_32 SDF directory under {dataroot}")

    return {
        "sdf_dir": sdf_dir,
        "lcd_path": next((path for path in lcd_candidates if path.exists()), None),
        "pld_path": next((path for path in pld_candidates if path.exists()), None),
        "vf_path": next((path for path in vf_candidates if path.exists()), None),
        "splits_dir": dataroot / "splits",
    }


def load_property_dict(path: Path, scale: float = 1.0):
    values = {}
    if path is None or not path.exists():
        return values
    with path.open("r") as handle:
        for line in handle:
            key, value = line.split()
            values[key] = float(value) * scale
    return values


def build_encoders_from_sdf_dir(sdf_dir: Path):
    topo_list = []
    node_list = ["N0"]
    edge_list = []
    for sdf_path in sdf_dir.glob("*.npy"):
        topo, node1, node2, edge = split_mof(sdf_path.stem)
        if topo not in topo_list:
            topo_list.append(topo)
        if node1 not in node_list:
            node_list.append(node1)
        if node2 not in node_list:
            node_list.append(node2)
        if edge not in edge_list:
            edge_list.append(edge)

    enc_topo = LabelEncoder().fit(topo_list)
    enc_node = LabelEncoder().fit(node_list)
    enc_edge = LabelEncoder().fit(edge_list)
    return {"topo": enc_topo, "node": enc_node, "edge": enc_edge}


class StandaloneMOFDataset(torch.utils.data.Dataset):
    def __init__(self, args: argparse.Namespace, phase: str, encoders):
        self.args = args
        self.phase = phase
        self.paths = resolve_paths(Path(args.dataroot))
        self.encoders = encoders
        self.sdf_dir = self.paths["sdf_dir"]
        split_path = self.paths["splits_dir"] / f"{phase}_split_{args.dataset_mode}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        with split_path.open("r") as handle:
            model_ids = [line.rstrip("\n") for line in handle]
        self.model_list = [self.sdf_dir / f"{model_id}.npy" for model_id in model_ids]
        self.model_list = self.model_list[: args.max_dataset_size]

        self.lcd_dict = load_property_dict(self.paths["lcd_path"], scale=0.01)
        self.pld_dict = load_property_dict(self.paths["pld_path"], scale=0.01)
        self.vf_dict = load_property_dict(self.paths["vf_path"], scale=1.0)

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, index):
        sdf_file = self.model_list[index]
        sdf = np.load(sdf_file).astype(np.float32)[:4]
        sdf = torch.tensor(sdf, dtype=torch.float32).view(4, self.args.res, self.args.res, self.args.res)
        if self.args.trunc_thres != 0.0:
            sdf[0, :] = torch.clamp(sdf[0, :], min=-self.args.trunc_thres, max=self.args.trunc_thres)

        mof_name = sdf_file.stem
        return {
            "sdf": sdf,
            "path": str(sdf_file),
            "id": torch.tensor(mof_to_id(self.encoders, mof_name), dtype=torch.long),
            "pld": self.pld_dict.get(mof_name, 0.0),
            "lcd": self.lcd_dict.get(mof_name, 0.0),
            "vf": self.vf_dict.get(mof_name, 0.0),
        }


def build_datasets(args: argparse.Namespace):
    paths = resolve_paths(Path(args.dataroot))
    encoders = build_encoders_from_sdf_dir(paths["sdf_dir"])
    train_ds = StandaloneMOFDataset(args, phase="train", encoders=encoders)
    test_ds = StandaloneMOFDataset(args, phase="test", encoders=encoders)

    if args.train_limit > 0:
        train_ds = Subset(train_ds, range(min(args.train_limit, len(train_ds))))
    if args.test_limit > 0:
        test_ds = Subset(test_ds, range(min(args.test_limit, len(test_ds))))

    return train_ds, test_ds, encoders


def prepare_input(sdf: torch.Tensor, input_variant: str, occ_source_channel: int) -> torch.Tensor:
    if input_variant == "sdf":
        return sdf
    occ = (sdf[:, occ_source_channel : occ_source_channel + 1] < 0).float()
    return torch.cat([sdf, occ], dim=1)


def build_model(args: argparse.Namespace, encoders) -> nn.Module:
    configs = omegaconf.OmegaConf.load(args.config)
    mparam = configs.model.params
    ddconfig = mparam.ddconfig
    in_channels = ddconfig.in_channels + (1 if args.input_variant == "sdf_occ" else 0)
    layers = mparam.layers
    kernel = mparam.kernel
    padding = mparam.padding

    topo_dim = len(encoders["topo"].classes_)
    node_dim = len(encoders["node"].classes_)
    edge_dim = len(encoders["edge"].classes_)

    if args.task == "topo":
        return build_topo_resnet("resnet18", TopoBasicBlock, layers, in_channels, kernel, padding, topo_dim)
    return build_bb_resnet("resnet18", BBBasicBlock, layers, in_channels, kernel, padding, topo_dim, node_dim, node_dim, edge_dim)


def evaluate(model, loader, args, device):
    model.eval()
    total = 0
    topo_correct = 0
    bb_total_correct = 0
    node1_correct = 0
    node2_correct = 0
    edge_correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            sdf = batch["sdf"].to(device)
            ids = batch["id"].to(device)
            x = prepare_input(sdf, args.input_variant, args.occ_source_channel)

            if args.task == "topo":
                logits = model(x)
                pred = logits.argmax(dim=-1)
                topo_correct += (pred == ids[:, 0]).sum().item()
                total += ids.size(0)
            else:
                node1_logits, node2_logits, edge_logits = model(x, ids[:, 0])
                node1_pred = node1_logits.argmax(dim=-1)
                node2_pred = node2_logits.argmax(dim=-1)
                edge_pred = edge_logits.argmax(dim=-1)
                node1_correct += (node1_pred == ids[:, 1]).sum().item()
                node2_correct += (node2_pred == ids[:, 2]).sum().item()
                edge_correct += (edge_pred == ids[:, 3]).sum().item()
                bb_total_correct += (
                    (node1_pred == ids[:, 1]) &
                    (node2_pred == ids[:, 2]) &
                    (edge_pred == ids[:, 3])
                ).sum().item()
                total += ids.size(0)

    if args.task == "topo":
        return {"topo_accuracy": topo_correct / max(total, 1)}
    return {
        "total_accuracy": bb_total_correct / max(total, 1),
        "node1_accuracy": node1_correct / max(total, 1),
        "node2_accuracy": node2_correct / max(total, 1),
        "edge_accuracy": edge_correct / max(total, 1),
    }


def train(model, train_loader, test_loader, args, device, save_dir: Path):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            sdf = batch["sdf"].to(device)
            ids = batch["id"].to(device)
            x = prepare_input(sdf, args.input_variant, args.occ_source_channel)

            optimizer.zero_grad(set_to_none=True)
            if args.task == "topo":
                logits = model(x)
                loss = criterion(logits, ids[:, 0])
            else:
                node1_logits, node2_logits, edge_logits = model(x, ids[:, 0])
                loss = (
                    criterion(node1_logits, ids[:, 1]) +
                    criterion(node2_logits, ids[:, 2]) +
                    criterion(edge_logits, ids[:, 3])
                )
            loss.backward()
            optimizer.step()

            batch_size = ids.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

        metrics = evaluate(model, test_loader, args, device)
        avg_loss = running_loss / max(seen, 1)
        key_metric = metrics["topo_accuracy"] if args.task == "topo" else metrics["total_accuracy"]
        record = {"epoch": epoch, "train_loss": avg_loss, **metrics}
        history.append(record)
        print(json.dumps(record))

        if key_metric > best_metric:
            best_metric = key_metric
            ckpt_path = save_dir / f"{args.task}_{args.input_variant}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                ckpt_path,
            )

    return history


def main():
    args = parse_args()
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_ds, test_ds, encoders = build_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    model = build_model(args, encoders).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])

    if args.eval_only:
        metrics = evaluate(model, test_loader, args, device)
        report = {
            "task": args.task,
            "input_variant": args.input_variant,
            "metrics": metrics,
            "checkpoint": args.checkpoint,
        }
    else:
        history = train(model, train_loader, test_loader, args, device, save_dir)
        report = {
            "task": args.task,
            "input_variant": args.input_variant,
            "best_checkpoint": str(save_dir / f"{args.task}_{args.input_variant}_best.pt"),
            "history": history,
        }

    report_path = save_dir / f"{args.task}_{args.input_variant}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
