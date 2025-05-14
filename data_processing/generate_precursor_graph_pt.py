"""
生成 precursor 的 graph 数据
[Data(x=[4, 200], edge_index=[2, 16], edge_attr=[16, 400], fc_weight=[4], comp_fea=[83]),
 Data(x=[3, 200], edge_index=[2, 9], edge_attr=[9, 400], fc_weight=[3], comp_fea=[83]),
 Data(x=[3, 200], edge_index=[2, 9], edge_attr=[9, 400], fc_weight=[3], comp_fea=[83]),
 ...]
"""

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
precursor_ids_path = f"raw/{dataset_name}_precursor_id.json"

embedding_path = "raw/matscholar.json"

# ========= 加载元素信息 =========
element_list = ["Cs", "K", "Rb", "Ba", "Na", "Sr", "Li", "Ca", "La", "Tb", "Yb", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Y", "Ho", "Er", "Tm", "Lu", "Pu", "Am", "Cm", "Hf", "Th", "Mg", "Zr", "Np", "Sc", "U", "Ta", "Ti", "Mn", "Be", "Nb", "Al", "Tl", "V", "Zn", "Cr", "Cd", "In", "Ga", "Fe", "Co", "Cu", "Re", "Si", "Tc", "Ni", "Ag", "Sn", "Hg", "Ge", "Bi", "B", "Sb", "Te", "Mo", "As", "P", "H", "Ir", "Os", "Pd", "Ru", "Pt", "Rh", "Pb", "W", "Au", "C", "Se", "S", "I", "Br", "N", "Cl", "O", "F"]
element_index_map = {el: i for i, el in enumerate(element_list)}

# ========= 加载数据 =========
with open(precursor_ids_path) as f:
    precursor_json = json.load(f)
precursor_id_to_composition = {int(k): Composition(v[0]) for k, v in precursor_json.items()}
num_precursors = len(precursor_id_to_composition)

with open(embedding_path) as f:
    element_to_embedding = json.load(f)

# ========= 工具函数 =========
def get_comp_fea(comp):
    fea = torch.zeros(len(element_list))
    for el in comp.elements:
        symbol = el.symbol
        if symbol in element_index_map:
            fea[element_index_map[symbol]] = comp.get_atomic_fraction(el)
    return fea

def build_data(formula, precursor_ids_list=[], split_type=""):
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


    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        fc_weight=fc_weight,
        comp_fea=comp_fea,
    )


    return data

# ========= 构建所有样本 =========
precursor_data = []
for pid, formula_list in precursor_json.items():
    pid = int(pid)
    formula = formula_list[0]  # 取列表中的第一个元素作为化学式
    precursor_data.append(build_data(formula))

# ========= 保存为 .pt 文件 =========
print(f"保存至 proceed/{dataset_name}_precursor_graph.pt")
torch.save(precursor_data, f"proceed/{dataset_name}_precursor_graph.pt")
print(f"✅ 保存完成：{len(precursor_data)}") 