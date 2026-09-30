# Bitplane Lab — CNN-DA-EMD-OLSB study guide

A beginner-friendly, interactive guide to the **CNN-Guided Distortion-Aware Adaptive EMD-OLSB Framework for Reversible Data Hiding in RGB Images** (B.Tech CSE 7th semester project, Parala Maharaja Engineering College, Berhampur, 2026–27).

**Live site:** https://omkar6060.github.io/cnn-da-emd-olsb-guide/

## What's inside
- `index.html` — the whole site (11 chapters, 3D bitplane stack, 3D distortion heightmap, EMD / OLSB / PSNR / capacity calculators). No build step.
- `starter/rdh_starter.py` — tested Python implementation: upper-bitplane map, Sobel + variance scoring, 3-class routing, boundary-safe mod-5 EMD, 3-bit OLSB, zlib location map, AES-256-GCM + PBKDF2, 64-byte header, exact recovery.

## Run the starter
```bash
pip install numpy pillow cryptography
python starter/rdh_starter.py
```
Expected: `secret ok: True`, `exact recovery: True`, `upper bits same: True`.

## Important
Read chapter 8 (Capacity reality check): on real photos the location map can need more room than the image offers. Discuss the suggested fixes with the project guides.
