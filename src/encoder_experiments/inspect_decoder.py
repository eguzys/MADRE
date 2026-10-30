import torch, importlib.util, inspect, sys, os

# load DeepSetsAE from the sweep script already on Euler
spec = importlib.util.spec_from_file_location(
    "ce", "/cluster/scratch/eguzys/compare_encoders_fixed.py")
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

AE = ce.DeepSetsAE
print("=== DeepSetsAE.__init__ signature ===")
print(inspect.signature(AE.__init__))
print("\n=== forward source ===")
print(inspect.getsource(AE.forward))
print("\n=== decoder attribute? ===")
m = AE(dim=6, hidden=64, output=8, max_n=22)
print([n for n,_ in m.named_children()])

# try a decode and report the shape the decoder emits
ckpt = torch.load("/cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt",
                  map_location="cpu")
sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
try:
    m.load_state_dict(sd); print("\nloaded state_dict OK")
except Exception as e:
    print("\nload note:", e)

x = torch.randn(2, 22, 6); mask = torch.ones(2, 22, dtype=torch.bool)
with torch.no_grad():
    out = m(x, mask)
print("\nforward returns type:", type(out))
if isinstance(out, (tuple, list)):
    for i, o in enumerate(out):
        print(f"  out[{i}] shape:", getattr(o, "shape", None))
else:
    print("  out shape:", out.shape)
