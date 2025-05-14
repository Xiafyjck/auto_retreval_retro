import pandas as pd
import json
import torch
import itertools
import sys
from collections import defaultdict
from pymatgen.core import Composition, Element
from torch_geometric.data import Data

# ========= 获取命令行参数 =========
dataset_name = "ceder"  # 默认值
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
    print(f"使用数据集: {dataset_name}")

# ========= 配置路径 =========
synthesis_path = f"raw/{dataset_name}_split.csv"
precursor_ids_path = f"raw/{dataset_name}_precursor_id.json"

embedding_path = "raw/matscholar.json"

# ========= 加载元素信息 =========
element_list = ["Cs", "K", "Rb", "Ba", "Na", "Sr", "Li", "Ca", "La", "Tb", "Yb", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Y", "Ho", "Er", "Tm", "Lu", "Pu", "Am", "Cm", "Hf", "Th", "Mg", "Zr", "Np", "Sc", "U", "Ta", "Ti", "Mn", "Be", "Nb", "Al", "Tl", "V", "Zn", "Cr", "Cd", "In", "Ga", "Fe", "Co", "Cu", "Re", "Si", "Tc", "Ni", "Ag", "Sn", "Hg", "Ge", "Bi", "B", "Sb", "Te", "Mo", "As", "P", "H", "Ir", "Os", "Pd", "Ru", "Pt", "Rh", "Pb", "W", "Au", "C", "Se", "S", "I", "Br", "N", "Cl", "O", "F"]
element_index_map = {el: i for i, el in enumerate(element_list)}

# ========= 加载数据 =========
df = pd.read_csv(synthesis_path)

with open(precursor_ids_path) as f:
    precursor_json = json.load(f)
precursor_id_to_composition = {int(k): Composition(v[0]) for k, v in precursor_json.items()}
num_precursors = len(precursor_id_to_composition)

with open(embedding_path) as f:
    element_to_embedding = json.load(f)

# ========= 构建 formula → 所有 precursor_id 配方列表 =========
formula_to_precursors = defaultdict(list)
for _, row in df.iterrows():
    formula = row["target"]
    pids = list(map(int, row["precursor_ids"].split(',')))
    formula_to_precursors[formula].append(pids)

# ========= 工具函数 =========
def get_comp_fea(comp):
    fea = torch.zeros(len(element_list))
    for el in comp.elements:
        symbol = el.symbol
        if symbol in element_index_map:
            fea[element_index_map[symbol]] = comp.get_atomic_fraction(el)
    return fea

def build_data(formula, precursor_ids_list, split_type):
    comp = Composition(formula)
    symbols = [el.symbol for el in comp.elements]

    x = torch.stack([torch.tensor(element_to_embedding[s], dtype=torch.float) for s in symbols])
    fc_weight = torch.tensor([comp.get_atomic_fraction(Element(s)) for s in symbols], dtype=torch.float)

    n = len(symbols)
    edge_index = torch.tensor(list(itertools.product(range(n), repeat=2)), dtype=torch.long).T
    edge_attr = torch.cat([
        torch.cat([x[i], x[j]]).unsqueeze(0)
        for i, j in zip(edge_index[0], edge_index[1])
    ], dim=0)

    comp_fea = get_comp_fea(comp)

    y_lb_one = torch.zeros(num_precursors)
    valid_precursor_ids = set(precursor_id_to_composition.keys())
    for pid in precursor_ids_list:
        if pid in valid_precursor_ids:
            y_lb_one[pid] = 1

    # 构造 y_multiple（统一加，不区分 train/val/test）
    y_multiple_list = []
    for pids in formula_to_precursors[formula]:
        y_multi = torch.zeros(num_precursors)
        for pid in pids:
            if pid in valid_precursor_ids:
                y_multi[pid] = 1
        y_multiple_list.append(y_multi)
    y_multiple = torch.stack(y_multiple_list)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        fc_weight=fc_weight,
        comp_fea=comp_fea,
        y_lb_one=y_lb_one,
        y_multiple=y_multiple,
        y_multiple_len=len(y_multiple_list),
    )

    data.split = split_type

    return data

# ========= 构建所有样本并划分 =========
train_data, val_data, test_data = [], [], []

for _, row in df.iterrows():
    formula = row['target']
    pids = list(map(int, row['precursor_ids'].split(',')))
    split = row['type']
    try:
        data = build_data(formula, pids, split)
        if split == "train":
            train_data.append(data)
        elif split == "val":
            val_data.append(data)
        elif split == "test":
            test_data.append(data)
    except Exception as e:
        print(f"[跳过] {formula} 错误：{e}")

# ========= 保存为 .pt 文件 =========
print(f"保存至 proceed/{dataset_name}_train_mpc.pt")
torch.save(train_data, f"proceed/{dataset_name}_train_mpc.pt")
print(f"保存至 proceed/{dataset_name}_val_mpc.pt")
torch.save(val_data, f"proceed/{dataset_name}_val_mpc.pt")
print(f"保存至 proceed/{dataset_name}_test_mpc.pt")
torch.save(test_data, f"proceed/{dataset_name}_test_mpc.pt")

print(f"✅ 保存完成：train={len(train_data)} val={len(val_data)} test={len(test_data)}")