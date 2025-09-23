import pmcx
import json

cfg = pmcx.io.json2mcx("test.json")
print(cfg)
print(pmcx.gpuinfo())
data = pmcx.mcxlab(cfg)
print(data)
