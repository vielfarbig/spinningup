import os
import torch

model = None

model_path = "../../../../data/ddpg/ddpg_s0/pyt_save/model.pt"
if os.path.isfile(model_path):
    ac = torch.load(model_path)
    model = ac.pi.pi
    torch.save(model, "../../../../data/ddpg/ddpg_s0/pyt_save/model_ac_pi_pi.pt")


