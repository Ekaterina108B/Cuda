from PIL import Image
import struct
import sys

in_path  = sys.argv[1]   # например, output.bin
out_path = sys.argv[2]   # например, output.png

with open(in_path, "rb") as f:
    # Заголовок: два int32 little-endian
    W = struct.unpack("<i", f.read(4))[0]
    H = struct.unpack("<i", f.read(4))[0]
    # Пиксели: W * H * 4 байта RGBA
    data = f.read(W * H * 4)

img = Image.frombytes("RGBA", (W, H), data)
img.save(out_path)
print(f"Saved {W}x{H} to {out_path}")