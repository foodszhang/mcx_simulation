import json


config = {}

from simple_gen import gen_shape, gen_volume_and_media

volfile, shapes, media = gen_volume_and_media("brain")


Domain = {
    # 二进制体素文件, 体素值与media对应，或者json定义的shapes文件
    # "VolumeFile": 'volume.bin',
    "VolumeFile": volfile,
    "Dim": shapes,
    "OriginType": 1,
    # 一个体素对应实际距离， 单位(mm)
    "LengthUnit": 0.1,
    "Media": media,
}
Session = {
    # 光子数
    "Photons": int(1e6),
    # 随机数种子
    "RNGSeed": 42,
    "ID": "test",
}
Forward = {
    "T0": 0.0e00,
    "T1": 5.0e-09,
    "DT": 5.0e-09,
}


radius = 5

sphsrc, shape = gen_shape(radius, "sphere")
# print("55555", sphsrc.shape)
sphsrc.tofile("test_source.bin")
Optode = {
    "Source": {
        "Pos": [113, 217, 118],
        "Dir": [0, 0, 1],
        "Type": "pattern3d",
        # 光源维度
        "Pattern": {
            "Nx": sphsrc.shape[0],
            "Ny": sphsrc.shape[1],
            "Nz": sphsrc.shape[2],
            "Data": "test_source.bin",
        },
        # 光源在维度下的分布， 值代表权重
        "Param1": shape,
    }
}
config["Domain"] = Domain
config["Session"] = Session
config["Forward"] = Forward
config["Optode"] = Optode

if __name__ == "__main__":
    with open("test.json", "w") as f:
        json.dump(config, f)
