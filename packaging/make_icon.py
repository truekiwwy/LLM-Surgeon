"""Draws the application icon (three stacked weight layers cut by a diagonal incision) with
the standard library only, and writes assets/icon.png and assets/icon.ico.

    python packaging/make_icon.py
"""
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "crates" / "surgeon-gui" / "assets"
BG, EDGE = (0x13, 0x16, 0x1B), (0x2A, 0x30, 0x39)
BARS = [((0x3F, 0xB8, 0xD4), 66), ((0x87, 0xAF, 0xFF), 110), ((0xFF, 0xAF, 0x00), 154)]
CUT = (0xD7, 0x5F, 0xD7)


def in_round_rect(x, y, r):
    cx, cy = min(max(x, r), 256 - r), min(max(y, r), 256 - r)
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def dist_to_line(x, y):
    # incision from (36, 206) to (220, 50)
    ax, ay, bx, by = 36, 206, 220, 50
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return ((x - (ax + t * dx)) ** 2 + (y - (ay + t * dy)) ** 2) ** 0.5


def sample(x, y):
    """RGBA of the icon at a point of the 256x256 design, or transparent."""
    if not in_round_rect(x, y, 52):
        return (0, 0, 0, 0)
    color = BG
    if not in_round_rect(x, y, 49):
        color = EDGE
    for c, top in BARS:
        if 52 <= x <= 204 and top <= y <= top + 30:
            color = c
    d = dist_to_line(x, y)
    if d <= 5:
        color = BG
    elif d <= 8:
        color = CUT
    return (*color, 255)


def render(size, ss=3):
    scale = 256 / size
    rows = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            acc = [0, 0, 0, 0]
            for sy in range(ss):
                for sx in range(ss):
                    r, g, b, a = sample((px + (sx + 0.5) / ss) * scale, (py + (sy + 0.5) / ss) * scale)
                    acc[0] += r * a; acc[1] += g * a; acc[2] += b * a; acc[3] += a
            n = ss * ss
            a = acc[3]
            row += bytes([acc[0] // a, acc[1] // a, acc[2] // a, a // n]) if a else b"\0\0\0\0"
        rows.append(bytes(row))
    return rows


def png(size):
    raw = b"".join(b"\0" + r for r in render(size))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def ico(sizes):
    images = [(s, png(s)) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for s, data in images:
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset + len(blobs))
        blobs += data
    return header + entries + blobs


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icon.png").write_bytes(png(256))
    (OUT / "icon.ico").write_bytes(ico([16, 32, 48, 64, 128, 256]))
    print("wrote", OUT / "icon.png", OUT / "icon.ico")
