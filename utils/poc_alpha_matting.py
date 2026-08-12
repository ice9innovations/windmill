#!/usr/bin/env python3
"""
Alpha matting PoC benchmark.

Fetches the _subject mask for a given image_id from sam3_results,
decodes it, generates a trimap, runs pymatting closed-form alpha
estimation, and saves the result. Times each stage.

Usage:
    python utils/poc_alpha_matting.py --image /path/to/image.png --image-id 318
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
import psycopg2
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


def connect():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        sslmode=os.getenv('DB_SSLMODE', 'prefer'),
    )


def fetch_subject_mask(conn, image_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT data->'_subject' FROM sam3_results WHERE image_id = %s",
        (image_id,)
    )
    row = cur.fetchone()
    cur.close()
    if not row or not row[0]:
        raise ValueError(f"No _subject mask found for image_id {image_id}")
    return row[0]


def decode_rle(rle, h, w):
    """Decode False-first RLE to boolean mask of shape (h, w)."""
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for count in rle:
        flat[pos:pos + count] = val
        pos += count
        val = not val
    return flat.reshape(h, w)


def union_masks(instances):
    """Union all instance masks into a single boolean mask."""
    first = instances[0]
    h, w = first['mask_shape']
    union = np.zeros(h * w, dtype=bool)
    for inst in instances:
        rle = inst.get('mask_rle', [])
        shape = inst.get('mask_shape', [h, w])
        if not rle or shape != [h, w]:
            continue
        flat = np.zeros(h * w, dtype=bool)
        pos, val = 0, False
        for count in rle:
            flat[pos:pos + count] = val
            pos += count
            val = not val
        union |= flat
    return union.reshape(h, w), h, w


def make_trimap(mask, erosion_px=15, dilation_px=15):
    """
    Generate a trimap from a binary mask.
    - Eroded interior  → definite foreground (1.0)
    - Dilated exterior → definite background (0.0)
    - Band between     → unknown (0.5)
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    struct = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), dtype=bool)
    eroded   = binary_erosion(mask,  structure=struct)
    dilated  = binary_dilation(mask, structure=struct)
    trimap = np.full(mask.shape, 0.5)   # unknown
    trimap[eroded]   = 1.0              # definite foreground
    trimap[~dilated] = 0.0              # definite background
    return trimap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to source image')
    parser.add_argument('--image-id', type=int, required=True, help='sam3_results image_id')
    parser.add_argument('--erosion', type=int, default=15, help='Trimap erosion px (default 15)')
    parser.add_argument('--dilation', type=int, default=15, help='Trimap dilation px (default 15)')
    parser.add_argument('--out', default='/tmp/alpha_matte_poc.png', help='Output path')
    args = parser.parse_args()

    print(f"Connecting to database...")
    conn = connect()

    print(f"Fetching _subject mask for image_id {args.image_id}...")
    subject = fetch_subject_mask(conn, args.image_id)
    conn.close()

    instances = subject.get('instances', [])
    if not instances:
        print("ERROR: _subject has no instances")
        sys.exit(1)

    print(f"Decoding {len(instances)} instance mask(s)...")
    t0 = time.time()
    mask, mask_h, mask_w = union_masks(instances)
    print(f"  mask shape: {mask_h}x{mask_w}, "
          f"subject pixels: {mask.sum():,} / {mask_h * mask_w:,} "
          f"({mask.mean() * 100:.1f}%)")
    print(f"  decode: {time.time() - t0:.3f}s")

    print(f"Loading image from {args.image}...")
    img = Image.open(args.image).convert('RGB')
    orig_w, orig_h = img.size
    print(f"  original size: {orig_w}x{orig_h}")

    if (orig_h, orig_w) != (mask_h, mask_w):
        print(f"  resizing to match mask: {mask_w}x{mask_h}")
        img = img.resize((mask_w, mask_h), Image.LANCZOS)

    image_np = np.array(img, dtype=np.float64) / 255.0

    print(f"Generating trimap (erosion={args.erosion}px, dilation={args.dilation}px)...")
    t0 = time.time()
    trimap = make_trimap(mask, args.erosion, args.dilation)
    unknown_px = (trimap == 0.5).sum()
    print(f"  unknown zone: {unknown_px:,} pixels ({unknown_px / mask.size * 100:.1f}% of image)")
    print(f"  trimap: {time.time() - t0:.3f}s")

    def save_result(alpha, label, out_path, image_np, mask_w, mask_h):
        alpha_uint8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha_uint8)
        alpha_img.save(out_path)

        composite = Image.new('RGBA', (mask_w, mask_h), (0, 200, 0, 255))
        src = Image.fromarray((image_np * 255).astype(np.uint8), 'RGB').convert('RGBA')
        src.putalpha(Image.fromarray(alpha_uint8))
        composite = Image.alpha_composite(composite, src)
        composite.save(out_path.replace('.png', '_composite.png'))

        buf = io.BytesIO()
        alpha_img.save(buf, format='PNG')
        b64_kb = len(base64.b64encode(buf.getvalue())) / 1024
        print(f"  [{label}] alpha PNG: {buf.tell() / 1024:.1f}KB  base64: {b64_kb:.1f}KB  → {out_path}")

    # --- Closed-form alpha matting ---
    print("\nRunning closed-form alpha matting...")
    t0 = time.time()
    from pymatting import estimate_alpha_cf
    alpha_cf = estimate_alpha_cf(image_np, trimap)
    elapsed_cf = time.time() - t0
    print(f"  closed-form matting: {elapsed_cf:.3f}s")
    save_result(alpha_cf, 'matting', args.out, image_np, mask_w, mask_h)

    # --- Two-stage: SAM3 pre-mask → rembg ---
    print("\nRunning two-stage: SAM3 pre-mask → rembg...")
    import rembg

    # Stage 1: apply SAM3 binary mask — zero out definite background pixels
    t0 = time.time()
    image_rgba = Image.fromarray((image_np * 255).astype(np.uint8), 'RGB').convert('RGBA')
    pixels = np.array(image_rgba)
    pixels[~mask, 3] = 0          # background pixels → fully transparent
    premasked = Image.fromarray(pixels, 'RGBA')
    t_premask = time.time() - t0
    print(f"  pre-mask: {t_premask:.3f}s")

    # Stage 2: rembg on the pre-masked image
    t0 = time.time()
    buf_in = io.BytesIO()
    premasked.save(buf_in, format='PNG')
    result_bytes = rembg.remove(buf_in.getvalue())
    t_rembg = time.time() - t0
    print(f"  rembg: {t_rembg:.3f}s")

    result_img = Image.open(io.BytesIO(result_bytes)).convert('RGBA')
    alpha_rembg = np.array(result_img)[:, :, 3].astype(np.float64) / 255.0

    elapsed_twostage = t_premask + t_rembg
    save_result(alpha_rembg, 'twostage', args.out.replace('.png', '_twostage.png'), image_np, mask_w, mask_h)

    # --- Three-stage: SAM3 pre-mask → rembg → alpha matte ---
    print("\nRunning three-stage: SAM3 pre-mask → rembg → alpha matte...")

    # Use rembg alpha as the trimap source — tighter unknown zone than raw SAM3 mask
    t0 = time.time()
    rembg_binary = alpha_rembg > 0.5
    trimap_3stage = make_trimap(rembg_binary, args.erosion, args.dilation)
    unknown_px_3 = (trimap_3stage == 0.5).sum()
    print(f"  trimap from rembg alpha: {unknown_px_3:,} unknown px "
          f"({unknown_px_3 / mask.size * 100:.1f}% vs {unknown_px / mask.size * 100:.1f}% from SAM3 mask)")
    alpha_3stage = estimate_alpha_cf(image_np, trimap_3stage)
    elapsed_3stage = time.time() - t0
    print(f"  matte on rembg: {elapsed_3stage:.3f}s")
    save_result(alpha_3stage, '3stage', args.out.replace('.png', '_3stage.png'), image_np, mask_w, mask_h)

    print(f"\nSummary:")
    print(f"  closed-form matting:     {elapsed_cf:.3f}s")
    print(f"  two-stage rembg:         {elapsed_twostage:.3f}s")
    print(f"  three-stage rembg+matte: {elapsed_twostage + elapsed_3stage:.3f}s  "
          f"(rembg {t_rembg:.3f}s + matte {elapsed_3stage:.3f}s)")


if __name__ == '__main__':
    main()
