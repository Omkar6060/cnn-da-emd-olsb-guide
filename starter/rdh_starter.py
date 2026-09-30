"""
CNN-DA-EMD-OLSB — beginner starter implementation (analytical map only).

Everything here uses numpy + cryptography so a beginner can run it
without a GPU. The DistortionCNN plugs in later through `cnn_map`.

Run:  python rdh_starter.py
"""
import math
import os
import struct
import zlib

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

MAGIC = b"RDH1"
HEADER_BYTES = 64
HEADER_PIXELS = math.ceil(HEADER_BYTES * 8 / 3)   # header lives in Blue low bits


# ---------- Step 1: bits ----------
def upper_bits(img):
    """Keep the top 5 bits. Embedding never touches these."""
    return img & 0xF8


# ---------- Step 2: distortion map ----------
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SY = SX.T
BOX = np.ones((3, 3)) / 9.0


def conv3(x, k):
    p = np.pad(x, 1, mode="edge")
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * p[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return out


def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def analytical_map(up):
    maps = []
    for c in range(3):
        ch = up[:, :, c].astype(np.float64)
        grad = np.hypot(conv3(ch, SX), conv3(ch, SY))
        mean = conv3(ch, BOX)
        var = conv3(ch * ch, BOX) - mean * mean
        maps.append(0.5 * norm01(grad) + 0.5 * norm01(var))
    return np.stack(maps, axis=-1)


def distortion_score(img, cnn_map=None, gamma=0.6):
    """One score per pixel in [0, 1]. Quantized so embed and extract agree."""
    a = analytical_map(upper_bits(img))
    d = a if cnn_map is None else gamma * cnn_map + (1 - gamma) * a
    return np.round(d.mean(axis=2) * 1024) / 1024


def classify(score, t1, t2):
    return np.digitize(score, [t1, t2])          # 0 smooth, 1 moderate, 2 texture


# ---------- Step 3: EMD mod-5 ----------
def f(p1, p2):
    return (p1 + 2 * p2) % 5


def emd_embed(p1, p2, d):
    """Smallest change to (p1, p2) so f == d, staying inside each 8-value block."""
    need = (d - f(p1, p2)) % 5
    if need == 0:
        return p1, p2
    best = None
    for a in range(-(p1 & 7), 8 - (p1 & 7)):
        for b in range(-(p2 & 7), 8 - (p2 & 7)):
            if (a + 2 * b) % 5 == need:
                cost = a * a + b * b
                if best is None or cost < best[0]:
                    best = (cost, a, b)
    _, a, b = best
    return p1 + a, p2 + b


def bytes_to_base5(data):
    n = int.from_bytes(data, "big")
    count = math.ceil(len(data) * 8 / math.log2(5))
    digits = []
    for _ in range(count):
        n, r = divmod(n, 5)
        digits.append(r)
    return digits[::-1]


def base5_to_bytes(digits, nbytes):
    n = 0
    for d in digits:
        n = n * 5 + d
    return n.to_bytes(nbytes, "big")


# ---------- Step 4: 3-bit OLSB ----------
def bytes_to_triplets(data):
    bits = "".join(f"{b:08b}" for b in data)
    bits += "0" * (-len(bits) % 3)
    return [int(bits[i:i + 3], 2) for i in range(0, len(bits), 3)]


def triplets_to_bytes(trips, nbytes):
    bits = "".join(f"{t:03b}" for t in trips)[: nbytes * 8]
    return int(bits, 2).to_bytes(nbytes, "big") if nbytes else b""


# ---------- Step 5: encryption ----------
def derive_key(password, salt):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    return kdf.derive(password.encode())


# ---------- Step 6: slots ----------
def slots(img, t1, t2, cnn_map=None):
    """Pixel positions for EMD (class 0/1) and OLSB (class 2), header excluded."""
    h, w, _ = img.shape
    cls = classify(distortion_score(img, cnn_map), t1, t2).ravel()
    idx = np.arange(h * w)
    usable = idx >= HEADER_PIXELS
    emd = idx[usable & (cls < 2)]
    olsb = idx[usable & (cls == 2)]
    return emd, olsb


def plan(ct_len, n_emd, n_olsb):
    """Split ciphertext bytes between EMD and OLSB. Returns (emd_bytes, used_emd, used_olsb)."""
    emd_bytes = min(ct_len, int(n_emd * math.log2(5) // 8))
    while emd_bytes > 0 and len(bytes_to_base5(b"\x00" * emd_bytes)) > n_emd:
        emd_bytes -= 1
    rest = ct_len - emd_bytes
    used_emd = len(bytes_to_base5(b"\x00" * emd_bytes)) if emd_bytes else 0
    used_olsb = math.ceil(rest * 8 / 3)
    if used_olsb > n_olsb:
        raise ValueError(f"Not enough capacity: need {used_olsb} OLSB pixels, have {n_olsb}")
    return emd_bytes, used_emd, used_olsb


def low_bits_of(flat, emd_pos, olsb_pos):
    """Original lower 3 bits of every channel we will touch (header + R,G + B)."""
    head = flat[:HEADER_PIXELS, 2] & 7
    rg = (flat[emd_pos][:, :2] & 7).ravel()
    b = flat[olsb_pos, 2] & 7
    return np.concatenate([head, rg, b]).astype(np.uint8).tobytes()


# ---------- Embed ----------
def embed(cover, secret, password, t1=0.15, t2=0.35, cnn_map=None):
    flat = cover.reshape(-1, 3).astype(np.int64).copy()
    emd_pos, olsb_pos = slots(cover, t1, t2, cnn_map)
    salt, nonce = os.urandom(16), os.urandom(12)
    aes = AESGCM(derive_key(password, salt))

    used_emd, used_olsb = 0, 0
    for _ in range(20):                           # grow until the map fits
        locmap = zlib.compress(low_bits_of(flat, emd_pos[:used_emd], olsb_pos[:used_olsb]), 9)
        plain = struct.pack(">III", len(secret), used_emd, used_olsb) + secret + locmap
        ct = aes.encrypt(nonce, plain, None)
        emd_bytes, need_emd, need_olsb = plan(len(ct), len(emd_pos), len(olsb_pos))
        if need_emd <= used_emd and need_olsb <= used_olsb:
            break
        used_emd, used_olsb = max(used_emd, need_emd), max(used_olsb, need_olsb)
    else:
        raise ValueError("Location map did not converge — payload too large for this image")

    header = (MAGIC + bytes([1, 0]) + salt + nonce
              + struct.pack(">IIHH", len(ct), emd_bytes, int(t1 * 1000), int(t2 * 1000)))
    header = header.ljust(HEADER_BYTES, b"\x00")

    for i, t in enumerate(bytes_to_triplets(header)):
        flat[i, 2] = (flat[i, 2] & 0xF8) | t

    for pos, d in zip(emd_pos, bytes_to_base5(ct[:emd_bytes]) if emd_bytes else []):
        flat[pos, 0], flat[pos, 1] = emd_embed(int(flat[pos, 0]), int(flat[pos, 1]), d)

    for pos, t in zip(olsb_pos, bytes_to_triplets(ct[emd_bytes:])):
        flat[pos, 2] = (flat[pos, 2] & 0xF8) | t

    return flat.astype(np.uint8).reshape(cover.shape)


# ---------- Extract + recover ----------
def extract(stego, password, cnn_map=None):
    flat = stego.reshape(-1, 3).astype(np.int64).copy()
    header = triplets_to_bytes([int(v) & 7 for v in flat[:HEADER_PIXELS, 2]], HEADER_BYTES)
    if header[:4] != MAGIC:
        raise ValueError("No RDH header found")
    salt, nonce = header[6:22], header[22:34]
    ct_len, emd_bytes, t1k, t2k = struct.unpack(">IIHH", header[34:46])

    emd_pos, olsb_pos = slots(stego, t1k / 1000, t2k / 1000, cnn_map)   # same map as embed
    n_digits = len(bytes_to_base5(b"\x00" * emd_bytes)) if emd_bytes else 0
    digits = [f(int(flat[p, 0]), int(flat[p, 1])) for p in emd_pos[:n_digits]]
    part1 = base5_to_bytes(digits, emd_bytes) if emd_bytes else b""
    rest = ct_len - emd_bytes
    trips = [int(flat[p, 2]) & 7 for p in olsb_pos[: math.ceil(rest * 8 / 3)]]
    ct = part1 + triplets_to_bytes(trips, rest)

    plain = AESGCM(derive_key(password, salt)).decrypt(nonce, ct, None)   # fails if tampered
    secret_len, used_emd, used_olsb = struct.unpack(">III", plain[:12])
    secret = plain[12:12 + secret_len]
    low = np.frombuffer(zlib.decompress(plain[12 + secret_len:]), dtype=np.uint8).astype(np.int64)

    k = HEADER_PIXELS
    flat[:k, 2] = (flat[:k, 2] & 0xF8) | low[:k]
    rg = low[k:k + 2 * used_emd].reshape(-1, 2)
    flat[emd_pos[:used_emd], 0] = (flat[emd_pos[:used_emd], 0] & 0xF8) | rg[:, 0]
    flat[emd_pos[:used_emd], 1] = (flat[emd_pos[:used_emd], 1] & 0xF8) | rg[:, 1]
    k += 2 * used_emd
    flat[olsb_pos[:used_olsb], 2] = (flat[olsb_pos[:used_olsb], 2] & 0xF8) | low[k:k + used_olsb]
    return secret, flat.astype(np.uint8).reshape(stego.shape)


# ---------- Metrics ----------
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:256, 0:256]
    cover = np.stack([(x + y) // 2, 255 - y, x], axis=-1).astype(np.uint8)
    cover[128:, 128:] = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)  # textured corner

    secret = b"Hello from the CNN-DA-EMD-OLSB starter!"
    stego = embed(cover, secret, "pmec-2027")
    got, recovered = extract(stego, "pmec-2027")

    print("secret ok      :", got == secret)
    print("exact recovery :", np.array_equal(recovered, cover))
    print("upper bits same:", np.array_equal(upper_bits(cover), upper_bits(stego)))
    print(f"PSNR           : {psnr(cover, stego):.2f} dB")
