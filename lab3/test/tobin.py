from PIL import Image
import struct
import sys

in_path  = sys.argv[1]
out_path = sys.argv[2]

# Открываем и приводим к RGBA (4 канала, 8 бит на канал)
img = Image.open(in_path).convert("RGBA")
W, H = img.size

with open(out_path, "wb") as f:
    # Заголовок: два int32 little-endian
    f.write(struct.pack("<i", W))
    f.write(struct.pack("<i", H))
    # Пиксели: RGBA, построчно, слева направо, сверху вниз
    f.write(img.tobytes())