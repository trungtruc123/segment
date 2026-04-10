"""
Tách CBCT volume chứa nhiều răng thành các volume răng riêng lẻ.

Chiến lược:
    1. Load CBCT volume + label mask (label: 0=bg, 1=tooth, 2=canal)
    2. Áp dụng connected component analysis trên tooth mask (label=1)
       để phân biệt các răng riêng lẻ (mỗi răng là 1 component)
    3. Với mỗi component (mỗi răng):
        - Tính bounding box
        - Mở rộng thêm margin để chứa đầy đủ chân răng + mô xung quanh
        - Crop cả image và mask
        - Chỉ giữ lại tooth + canal thuộc răng đó (loại bỏ các răng khác
          có thể lọt vào bounding box)
    4. Lưu từng răng thành file .nii.gz riêng

Cách dùng:
    python split_teeth.py --image data/raw/SLZ000.nii.gz \
                          --label data/raw/SLZ000-label.nii.gz \
                          --output data/teeth/ \
                          --case_id SLZ000

    # Hoặc xử lý toàn bộ thư mục:
    python split_teeth.py --batch --input_dir data/raw --output data/teeth/
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load NIfTI file, trả về (data, affine, header)."""
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header,
    path: str,
    is_label: bool = False,
):
    """
    Lưu numpy array thành NIfTI.

    Args:
        is_label: nếu True, lưu dưới dạng uint8 (cho mask segmentation).
                  Nếu False, lưu float32 (cho CBCT image).
    """
    if is_label:
        # Mask phải là integer để 3D Slicer và MONAI đọc đúng
        data_out = data.astype(np.uint8)
        img = nib.Nifti1Image(data_out, affine=affine)
        img.header.set_data_dtype(np.uint8)
    else:
        data_out = data.astype(np.float32)
        img = nib.Nifti1Image(data_out, affine=affine, header=header)
        img.header.set_data_dtype(np.float32)
    nib.save(img, path)


def find_individual_teeth(
    label: np.ndarray,
    tooth_label: int = 1,
    canal_label: int = 2,
    min_voxels: int = 5000,
) -> Tuple[np.ndarray, int]:
    """
    Phân tách các răng riêng lẻ bằng connected component analysis.

    QUAN TRỌNG: Dùng UNION (tooth ∪ canal) làm "full tooth" mask để
    connected component không bị vỡ bởi cấu trúc canal bên trong.
    Nếu chỉ dùng label==1 (tooth shell), mask là lớp vỏ rỗng — CC
    có thể bị ảnh hưởng và sau đó việc filter canal bằng dilation sẽ
    bỏ sót phần pulp chamber ở giữa răng.

    Args:
        label: mask 3D (int)
        tooth_label: label của tooth shell (thường = 1)
        canal_label: label của canal (thường = 2)
        min_voxels: ngưỡng tối thiểu để loại bỏ noise / component nhỏ

    Returns:
        components: mảng 3D cùng shape với label, mỗi răng (full volume)
                    có 1 số nguyên index
        num_teeth: số lượng răng tìm được
    """
    # Union của tooth + canal = toàn bộ thể tích răng (bao gồm khoang tuỷ)
    full_tooth_mask = (
        (label == tooth_label) | (label == canal_label)
    ).astype(np.uint8)

    # Connectivity = 1 (6-connectivity) để tránh nối nhầm 2 răng
    # gần nhau qua điểm góc
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, num_found = ndimage.label(full_tooth_mask, structure=structure)

    # Lọc bỏ các component quá nhỏ (nhiễu)
    component_sizes = ndimage.sum(full_tooth_mask, labeled, range(1, num_found + 1))
    keep_mask = component_sizes >= min_voxels

    # Relabel: chỉ giữ components lớn
    new_labeled = np.zeros_like(labeled)
    new_idx = 1
    for i, keep in enumerate(keep_mask):
        if keep:
            new_labeled[labeled == (i + 1)] = new_idx
            new_idx += 1

    num_teeth = new_idx - 1
    return new_labeled, num_teeth


def get_bounding_box(
    mask: np.ndarray,
    margin: int = 10,
) -> Tuple[slice, slice, slice]:
    """
    Tính bounding box 3D của mask với margin.
    Trả về tuple các slice để có thể dùng trực tiếp: volume[bbox].
    """
    coords = np.array(np.where(mask > 0))
    if coords.size == 0:
        return None

    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1  # +1 vì slice không bao gồm end

    # Mở rộng margin nhưng clip trong biên volume
    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, mask.shape)

    return tuple(slice(min_coords[i], max_coords[i]) for i in range(3))


def extract_single_tooth(
    image: np.ndarray,
    label: np.ndarray,
    tooth_components: np.ndarray,
    tooth_idx: int,
    margin: int = 15,
    tooth_label: int = 1,
    canal_label: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trích xuất 1 răng riêng lẻ từ volume.

    tooth_components đã chứa full tooth (tooth shell + canal) như 1 khối,
    nên ta dùng trực tiếp thay vì phải dilation để bắt canal.

    Args:
        image: CBCT volume gốc
        label: mask gốc (0=bg, 1=tooth, 2=canal)
        tooth_components: mảng components từ find_individual_teeth
                          (chứa full tooth volume, không chỉ shell)
        tooth_idx: index của răng cần trích xuất (1, 2, ..., num_teeth)
        margin: margin voxel quanh bounding box
        tooth_label: label của tooth shell
        canal_label: label của canal

    Returns:
        cropped_image (float32), cropped_label (uint8)
    """
    # Full volume của răng hiện tại (tooth shell + canal bên trong)
    this_full_tooth = (tooth_components == tooth_idx)

    # Tính bounding box trên full volume → đảm bảo bao trọn cả crown
    # lẫn root apex lẫn pulp chamber
    bbox = get_bounding_box(this_full_tooth, margin=margin)
    if bbox is None:
        return None, None

    # Crop
    cropped_image = image[bbox].copy()
    cropped_label_full = label[bbox].copy()
    cropped_this_tooth = this_full_tooth[bbox]

    # Tạo label mới: chỉ giữ voxels thuộc răng hiện tại
    # (loại bỏ phần răng lân cận có thể lọt vào bbox)
    cropped_label = np.zeros_like(cropped_label_full, dtype=np.uint8)

    # Copy lại label gốc (cả tooth shell và canal) BÊN TRONG component
    # của răng hiện tại → không cần dilation, không mất voxels canal
    tooth_voxels = cropped_this_tooth & (cropped_label_full == tooth_label)
    canal_voxels = cropped_this_tooth & (cropped_label_full == canal_label)
    cropped_label[tooth_voxels] = tooth_label
    cropped_label[canal_voxels] = canal_label

    return cropped_image, cropped_label


def find_teeth_from_image(
    image: np.ndarray,
    percentile_low: float = 60.0,
    percentile_high: float = 99.5,
    min_voxels: int = 5000,
    seed_percentile: float = 85.0,
    num_expected: int = 6,
) -> Tuple[np.ndarray, int]:
    """
    Tìm các răng riêng lẻ từ CBCT image (không cần label).

    Chiến lược: Dual-mask + Distance Transform + peak_local_max + Watershed
    + post-merge regions nhỏ/gần nhau.

        Vấn đề: các răng nằm sát nhau, phần ngà răng (dentin) cùng
        intensity → threshold thông thường gộp 2-3 răng thành 1 khối.

        Giải pháp:
        1. Otsu → full_mask (toàn bộ cấu trúc răng + bone)
        2. tooth_mask (percentile 65 của foreground + opening mạnh):
           chỉ giữ lõi cứng nhất (enamel + dentin core), cắt dentin bridges
        3. Distance transform trên tooth_mask → peaks ở tâm mỗi răng
        4. peak_local_max + plateau scan → tìm N ổn định
        5. Watershed(-dist, seeds, full_mask) → mở rộng ra toàn biên
        6. Post-merge: nếu N > expected, merge regions nhỏ vào neighbor gần nhất
        7. Filter bỏ component nhỏ (noise)

    Args:
        image: CBCT volume (float) — có thể là bản đã downsample
        percentile_low: percentile ngưỡng dưới (fallback nếu Otsu fail)
        percentile_high: clip outlier trước khi threshold
        min_voxels: ngưỡng tối thiểu cho kết quả cuối (loại noise)
        seed_percentile: (deprecated — giữ cho backward compat)
        num_expected: số răng mong đợi (default=6), dùng cho fallback + merge

    Returns:
        components: mảng 3D, mỗi răng có 1 integer index
        num_teeth: số lượng răng tìm được
    """
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import gaussian_filter

    # =================================================================
    # 1. Normalize
    # =================================================================
    p_low = np.percentile(image, 0.5)
    p_high = np.percentile(image, percentile_high)
    img_norm = np.clip(image, p_low, p_high)
    img_norm = (img_norm - p_low) / (p_high - p_low + 1e-8)

    # =================================================================
    # 2. Dual mask: FULL mask (Otsu) + TOOTH mask (cao hơn, opening mạnh)
    # =================================================================
    try:
        from skimage.filters import threshold_otsu
        fg_vals = img_norm[img_norm > 0.01]
        otsu_t = threshold_otsu(fg_vals) if len(fg_vals) > 1000 else 0.3
    except ImportError:
        otsu_t = np.percentile(img_norm, percentile_low)

    # FULL mask: Otsu → bao gồm toàn bộ cấu trúc răng cho watershed
    full_mask = (img_norm >= otsu_t).astype(np.uint8)
    full_mask = ndimage.binary_fill_holes(full_mask).astype(np.uint8)

    # TOOTH mask: percentile 65 CỦA foreground — cao hơn median (50) để
    # loại alveolar bone (hơi tối hơn enamel+dentin), nhưng không quá cao
    # (75-80) sẽ mất chân răng nhỏ
    fg_intensities = img_norm[full_mask > 0]
    tooth_t = np.percentile(fg_intensities, 65)
    tooth_mask = (img_norm >= tooth_t).astype(np.uint8)
    tooth_mask = ndimage.binary_fill_holes(tooth_mask).astype(np.uint8)

    # Opening MẠNH trên tooth_mask: dùng connectivity=2 (18-connected, lớn hơn)
    # + 3 iterations để cắt dentin bridges mỏng giữa các răng
    struct_open = ndimage.generate_binary_structure(3, 2)  # 18-connectivity
    tooth_mask = ndimage.binary_opening(
        tooth_mask, structure=struct_open, iterations=3
    ).astype(np.uint8)

    # Nếu opening quá mạnh xoá hết, fallback: giảm iterations
    if tooth_mask.sum() < full_mask.sum() * 0.05:
        print(f"    [WARN] Opening quá mạnh, fallback iterations=1")
        tooth_mask = (img_norm >= tooth_t).astype(np.uint8)
        tooth_mask = ndimage.binary_fill_holes(tooth_mask).astype(np.uint8)
        struct_open_light = ndimage.generate_binary_structure(3, 1)
        tooth_mask = ndimage.binary_opening(
            tooth_mask, structure=struct_open_light, iterations=2
        ).astype(np.uint8)

    print(f"    Otsu: {otsu_t:.3f} → full_mask: {full_mask.sum():,} vx")
    print(f"    Tooth (p65): {tooth_t:.3f} → tooth_mask: {tooth_mask.sum():,} vx "
          f"(ratio={tooth_mask.sum()/max(full_mask.sum(),1)*100:.1f}%)")

    # =================================================================
    # 3. Distance Transform trên TOOTH mask + peak_local_max
    # =================================================================
    dist = ndimage.distance_transform_edt(tooth_mask)

    # Smooth: sigma=3.0 voxels (~0.75mm ở 0.25mm spacing)
    # Đủ để merge double-peaks trên 1 răng, giữ 2 peaks giữa 2 răng sát nhau
    sigma = 3.0
    dist_smooth = gaussian_filter(dist, sigma=sigma)
    max_dist = dist.max()
    print(f"    Distance transform: max={max_dist:.1f} vx, sigma={sigma:.1f}")

    if max_dist < 2.0:
        print(f"    [WARN] max_dist quá nhỏ ({max_dist:.1f}), không tìm được peaks")
        return np.zeros_like(image, dtype=np.int32), 0

    # threshold_abs: peak phải cách biên ≥ 20% max_distance (giảm từ 25%
    # để không bỏ sót răng nhỏ — sẽ dùng post-merge để loại noise thay thế)
    thresh_abs = max(2.5, max_dist * 0.20)

    # =================================================================
    # Scan min_distance: tìm PLATEAU cho N peaks ổn định
    # =================================================================
    EXPECTED_MIN = max(2, num_expected - 3)
    EXPECTED_MAX = num_expected + 4

    md_max = max(5, int(max_dist * 0.9))
    md_min = max(2, int(max_dist * 0.08))
    scan_results = []

    for md in range(md_max, md_min - 1, -1):
        coords_tmp = peak_local_max(
            dist_smooth,
            min_distance=md,
            threshold_abs=thresh_abs,
            exclude_border=False,
        )
        n = len(coords_tmp)
        scan_results.append((md, n, coords_tmp))

    # Log scan results
    unique_ns = []
    for md, n, _ in scan_results:
        if not unique_ns or unique_ns[-1][1] != n:
            unique_ns.append((md, n))
    print(f"    Scan: {', '.join(f'md={md}→{n}' for md, n in unique_ns[:15])}")

    # --- Tìm plateau: N giữ nguyên qua ≥3 bước liên tiếp ---
    best_md = None
    best_n = 0
    best_coords = None

    # Pass 1: tìm plateau CHÍNH XÁC num_expected
    for i in range(len(scan_results) - 2):
        md_i, n_i, c_i = scan_results[i]
        _, n_i1, _ = scan_results[i + 1]
        _, n_i2, _ = scan_results[i + 2]
        if n_i == num_expected and n_i == n_i1 == n_i2:
            best_md, best_n, best_coords = md_i, n_i, c_i
            break

    # Pass 2: tìm plateau bất kỳ trong [EXPECTED_MIN, EXPECTED_MAX]
    if best_coords is None:
        for i in range(len(scan_results) - 2):
            md_i, n_i, c_i = scan_results[i]
            _, n_i1, _ = scan_results[i + 1]
            _, n_i2, _ = scan_results[i + 2]
            if (EXPECTED_MIN <= n_i <= EXPECTED_MAX
                    and n_i == n_i1 == n_i2):
                best_md, best_n, best_coords = md_i, n_i, c_i
                break

    # Pass 3: tìm plateau ngắn hơn (≥2 bước liên tiếp)
    if best_coords is None:
        for i in range(len(scan_results) - 1):
            md_i, n_i, c_i = scan_results[i]
            _, n_i1, _ = scan_results[i + 1]
            if (EXPECTED_MIN <= n_i <= EXPECTED_MAX
                    and n_i == n_i1):
                best_md, best_n, best_coords = md_i, n_i, c_i
                break

    # Fallback: chọn N gần num_expected nhất
    if best_coords is None:
        valid = [(md, n, c) for md, n, c in scan_results
                 if EXPECTED_MIN <= n <= EXPECTED_MAX]
        if valid:
            # Ưu tiên: |N - num_expected| nhỏ nhất, sau đó md lớn nhất (ít peaks hơn = sạch hơn)
            valid.sort(key=lambda x: (abs(x[1] - num_expected), -x[0]))
            best_md, best_n, best_coords = valid[0]
        else:
            # Không có gì trong range → chọn N nhỏ nhất ≥ 2
            valid_any = [(md, n, c) for md, n, c in scan_results if n >= 2]
            if valid_any:
                valid_any.sort(key=lambda x: (abs(x[1] - num_expected), -x[0]))
                best_md, best_n, best_coords = valid_any[0]

    print(f"    → Best: min_distance={best_md}, peaks={best_n}, "
          f"thresh_abs={thresh_abs:.1f}, expected={num_expected}")

    if best_coords is None or best_n == 0:
        return np.zeros_like(image, dtype=np.int32), 0

    coords = best_coords
    n_seeds = best_n

    # =================================================================
    # 4. Tạo seed markers + Watershed trên FULL mask
    # =================================================================
    seed_labels = np.zeros(full_mask.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, 1):
        seed_labels[z, y, x] = i
    # Dilate seeds nhẹ (3³ cube) để watershed khởi đầu ổn định
    seed_labels = ndimage.maximum_filter(seed_labels, size=3)

    # Watershed: dùng -dist (valley ở biên giữa 2 răng) trên full_mask
    labeled = watershed(-dist, markers=seed_labels, mask=full_mask)

    # =================================================================
    # 5. Post-merge: nếu quá nhiều regions, merge nhỏ vào neighbor gần nhất
    # =================================================================
    max_label = labeled.max()
    if max_label == 0:
        return np.zeros_like(image, dtype=np.int32), 0

    # Tính size và centroid cho mỗi region
    region_sizes = np.array([
        int((labeled == lbl).sum()) for lbl in range(1, max_label + 1)
    ])
    region_centroids = np.array([
        np.mean(np.argwhere(labeled == lbl), axis=0)
        for lbl in range(1, max_label + 1)
    ])

    # Merge strategy 2 bước:
    #   Bước A: merge region NHỎ (< 30% median) vào neighbor gần nhất
    #   Bước B: nếu vẫn N > num_expected → merge 2 regions có centroid
    #           GẦN NHAU NHẤT (= 2 nửa cùng 1 răng bị watershed tách nhầm)
    merge_map = {i: i for i in range(max_label)}

    def get_valid_idxs():
        return [i for i in range(max_label)
                if region_sizes[i] >= min_voxels and merge_map[i] == i]

    def do_merge(src, dst, reason=""):
        """Gộp region src vào dst."""
        labeled[labeled == (src + 1)] = dst + 1
        old_sz = region_sizes[src]
        w1, w2 = region_sizes[dst], region_sizes[src]
        region_centroids[dst] = (
            region_centroids[dst] * w1 + region_centroids[src] * w2
        ) / (w1 + w2 + 1e-8)
        region_sizes[dst] += region_sizes[src]
        region_sizes[src] = 0
        merge_map[src] = dst
        n_now = len(get_valid_idxs())
        print(f"    [Merge-{reason}] region {src+1} ({old_sz:,}vx) "
              f"→ region {dst+1}, remaining={n_now}")

    # --- Bước A: merge regions nhỏ ---
    median_size = (np.median(region_sizes[region_sizes >= min_voxels])
                   if np.any(region_sizes >= min_voxels)
                   else np.median(region_sizes))
    small_threshold = median_size * 0.30

    while len(get_valid_idxs()) > num_expected:
        valid_idxs = get_valid_idxs()
        smallest_idx = min(valid_idxs, key=lambda i: region_sizes[i])
        if region_sizes[smallest_idx] > small_threshold:
            break  # Không còn region nhỏ → chuyển sang bước B

        # Tìm neighbor gần nhất
        nearest_idx = min(
            (j for j in valid_idxs if j != smallest_idx),
            key=lambda j: np.linalg.norm(
                region_centroids[smallest_idx] - region_centroids[j]),
            default=-1,
        )
        if nearest_idx < 0:
            break
        do_merge(smallest_idx, nearest_idx, reason="small")

    # --- Bước B: force-merge 2 regions gần nhau nhất (proximity) ---
    # Khi tất cả regions đều lớn nhưng N vẫn > expected,
    # 2 regions gần nhất rất có thể là 1 răng bị tách đôi
    while len(get_valid_idxs()) > num_expected:
        valid_idxs = get_valid_idxs()
        if len(valid_idxs) <= num_expected:
            break

        # Tìm cặp (i, j) có centroid distance nhỏ nhất
        best_pair = None
        best_dist_val = float("inf")
        for ii, idx_a in enumerate(valid_idxs):
            for idx_b in valid_idxs[ii + 1:]:
                d = np.linalg.norm(
                    region_centroids[idx_a] - region_centroids[idx_b])
                if d < best_dist_val:
                    best_dist_val = d
                    best_pair = (idx_a, idx_b)

        if best_pair is None:
            break

        # Merge region nhỏ hơn vào region lớn hơn
        a, b = best_pair
        if region_sizes[a] < region_sizes[b]:
            do_merge(a, b, reason="proximity")
        else:
            do_merge(b, a, reason="proximity")

    # =================================================================
    # 6. Filter bỏ component nhỏ + relabel liên tục
    # =================================================================
    unique_labels = sorted(set(labeled[labeled > 0]))
    new_labeled = np.zeros_like(labeled)
    new_idx = 1
    for lbl in unique_labels:
        sz = int((labeled == lbl).sum())
        if sz >= min_voxels:
            new_labeled[labeled == lbl] = new_idx
            new_idx += 1

    num_teeth = new_idx - 1
    print(f"    Watershed → {max_label} regions → merge → "
          f"filter (min={min_voxels}): {num_teeth} răng")
    return new_labeled, num_teeth


def process_case_inference(
    image_path: str,
    output_dir: str,
    case_id: str,
    margin: int = 15,
    min_voxels: int = 5000,
    percentile_threshold: float = 60.0,
) -> int:
    """
    Tách CBCT thành các răng riêng lẻ CHỈ TỪ IMAGE (không cần label).
    Dùng cho inference trên data mới chưa có annotation.

    Returns:
        Số lượng răng đã lưu
    """
    print(f"\n[{case_id}] Loading {image_path}")
    image, affine, header = load_nifti(image_path)
    print(f"  Volume shape: {image.shape}")

    components, num_teeth = find_teeth_from_image(
        image,
        percentile_low=percentile_threshold,
        min_voxels=min_voxels,
    )
    print(f"  Tìm thấy {num_teeth} răng")

    if num_teeth == 0:
        print(f"  [WARN] Không tìm thấy răng nào trong {case_id}")
        return 0

    out_img_dir = Path(output_dir) / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for tooth_idx in range(1, num_teeth + 1):
        this_tooth = (components == tooth_idx)
        bbox = get_bounding_box(this_tooth, margin=margin)
        if bbox is None:
            continue

        cropped_img = image[bbox].copy()
        out_name = f"{case_id}_tooth{tooth_idx:02d}.nii.gz"
        img_out = out_img_dir / out_name
        save_nifti(cropped_img, affine, header, str(img_out), is_label=False)
        print(f"  Răng {tooth_idx:02d}: shape={cropped_img.shape} -> {out_name}")

    return num_teeth


def process_case(
    image_path: str,
    label_path: str,
    output_dir: str,
    case_id: str,
    margin: int = 15,
    min_voxels: int = 5000,
) -> int:
    """
    Xử lý 1 ca CBCT: tách thành các file răng riêng lẻ.

    Returns:
        Số lượng răng đã được lưu
    """
    print(f"\n[{case_id}] Loading {image_path}")
    image, affine, header = load_nifti(image_path)
    label, _, label_header = load_nifti(label_path)

    print(f"  Volume shape: {image.shape}")
    print(f"  Label unique values: {np.unique(label.astype(int))}")

    # Phân tách các răng riêng lẻ
    components, num_teeth = find_individual_teeth(
        label.astype(int), tooth_label=1, min_voxels=min_voxels
    )
    print(f"  Tìm thấy {num_teeth} răng")

    if num_teeth == 0:
        print(f"  [WARN] Không tìm thấy răng nào trong {case_id}")
        return 0

    # Tạo thư mục output
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Trích xuất và lưu từng răng
    for tooth_idx in range(1, num_teeth + 1):
        cropped_img, cropped_lbl = extract_single_tooth(
            image, label, components, tooth_idx,
            margin=margin, canal_label=2,
        )

        if cropped_img is None:
            continue

        # Thống kê
        n_tooth = int((cropped_lbl == 1).sum())
        n_canal = int((cropped_lbl == 2).sum())
        has_canal = "OK" if n_canal > 0 else "NO_CANAL"

        # Đặt tên: <case_id>_tooth<idx>.nii.gz
        out_name = f"{case_id}_tooth{tooth_idx:02d}.nii.gz"
        img_out = out_img_dir / out_name
        lbl_out = out_lbl_dir / out_name

        save_nifti(cropped_img, affine, header, str(img_out), is_label=False)
        save_nifti(cropped_lbl, affine, label_header, str(lbl_out), is_label=True)

        print(
            f"  Răng {tooth_idx:02d}: shape={cropped_img.shape} "
            f"tooth={n_tooth:,}vx canal={n_canal:,}vx [{has_canal}] -> {out_name}"
        )

    return num_teeth


def batch_process(input_dir: str, output_dir: str, margin: int, min_voxels: int):
    """Tự động tìm cặp image/label trong input_dir và xử lý toàn bộ."""
    input_path = Path(input_dir)

    # Tìm tất cả file image (không chứa 'label' trong tên)
    all_files = sorted(list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii")))
    image_files = [f for f in all_files if "label" not in f.name.lower()
                   and "mask" not in f.name.lower() and "seg" not in f.name.lower()]

    print(f"Tìm thấy {len(image_files)} file image trong {input_dir}")

    total_teeth = 0
    for img_file in image_files:
        stem = img_file.name.replace(".nii.gz", "").replace(".nii", "")
        # Tìm label tương ứng (pattern: <stem>-label.nii.gz hoặc <stem>_label.nii.gz)
        candidates = [
            input_path / f"{stem}-label.nii.gz",
            input_path / f"{stem}_label.nii.gz",
            input_path / f"{stem}-mask.nii.gz",
            input_path / f"{stem}_seg.nii.gz",
        ]
        label_file = next((c for c in candidates if c.exists()), None)
        if label_file is None:
            print(f"[SKIP] Không tìm thấy label cho {img_file.name}")
            continue

        n = process_case(
            str(img_file), str(label_file), output_dir, case_id=stem,
            margin=margin, min_voxels=min_voxels,
        )
        total_teeth += n

    print(f"\n=== HOÀN TẤT ===")
    print(f"Tổng số răng đã trích xuất: {total_teeth}")
    print(f"Lưu tại: {output_dir}/images và {output_dir}/masks")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tách CBCT volume thành các răng riêng lẻ"
    )
    parser.add_argument("--image", type=str, help="Đường dẫn file CBCT")
    parser.add_argument("--label", type=str, help="Đường dẫn file label")
    parser.add_argument("--output", type=str, default="./data/teeth",
                        help="Thư mục output")
    parser.add_argument("--case_id", type=str, default="case",
                        help="ID của ca (dùng làm prefix cho file output)")
    parser.add_argument("--margin", type=int, default=15,
                        help="Margin (voxel) quanh bounding box của răng")
    parser.add_argument("--min_voxels", type=int, default=5000,
                        help="Số voxel tối thiểu để coi là 1 răng (lọc noise)")
    parser.add_argument("--batch", action="store_true",
                        help="Xử lý toàn bộ thư mục")
    parser.add_argument("--input_dir", type=str,
                        help="Thư mục chứa các file CBCT (dùng với --batch)")
    parser.add_argument("--infer_only", action="store_true",
                        help="Chỉ có image, không có label (dùng cho inference)")
    parser.add_argument("--percentile_threshold", type=float, default=60.0,
                        help="Percentile ngưỡng phân vùng răng (default=60, tăng nếu có nhiều noise)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.infer_only:
        # Inference mode: không cần label
        if args.batch:
            if not args.input_dir:
                raise ValueError("--batch yêu cầu --input_dir")
            input_path = Path(args.input_dir)
            all_files = sorted(
                list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
                + list(input_path.glob("*.nrrd"))
            )
            image_files = [f for f in all_files
                           if "label" not in f.name.lower()
                           and "mask" not in f.name.lower()
                           and "seg" not in f.name.lower()
                           and "pred" not in f.name.lower()]
            total = 0
            for img_file in image_files:
                stem = img_file.name.replace(".nii.gz", "").replace(".nii", "").replace(".nrrd", "")
                total += process_case_inference(
                    str(img_file), args.output, stem,
                    margin=args.margin, min_voxels=args.min_voxels,
                    percentile_threshold=args.percentile_threshold,
                )
            print(f"\nTổng: {total} răng -> {args.output}/images/")
        else:
            if not args.image:
                raise ValueError("--infer_only yêu cầu --image")
            process_case_inference(
                args.image, args.output, args.case_id,
                margin=args.margin, min_voxels=args.min_voxels,
                percentile_threshold=args.percentile_threshold,
            )
    elif args.batch:
        if not args.input_dir:
            raise ValueError("--batch yêu cầu --input_dir")
        batch_process(args.input_dir, args.output, args.margin, args.min_voxels)
    else:
        if not args.image or not args.label:
            raise ValueError("Cần --image và --label (hoặc dùng --batch hoặc --infer_only)")
        process_case(
            args.image, args.label, args.output, args.case_id,
            margin=args.margin, min_voxels=args.min_voxels,
        )


if __name__ == "__main__":
    main()
