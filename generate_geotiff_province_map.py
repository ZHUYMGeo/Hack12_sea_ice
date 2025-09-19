import pdb
import torch
import numpy as np
import yaml
import os
import rasterio
from collections import OrderedDict
from tqdm import tqdm
import traceback
import time
from osgeo import gdal, osr
from skimage.util import img_as_float
from datasketch import HyperLogLog
import seaborn as sns


def pot_CM(save_dir, confusion_matrix, confusion_classes):
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # Plot raw counts
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='.2%', cmap='Blues', cbar_kws={'label': 'Count'},
                     xticklabels=confusion_classes, yticklabels=confusion_classes)

    plt.title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir + '/confusion_matrix2.png')


def write_classified_geotiff_with_colormap(output_path, Classified_Scene, Uncertainty_Scene, uncertainty_map_path,
                                           gdal_transform, projection_wkt):
    """
    Write classification results to GeoTIFF with embedded color table
    """
    # Get driver
    driver = gdal.GetDriverByName('GTiff')

    # Set up the output dataset with LZW compression
    options = [
        'COMPRESS=LZW',
        'PREDICTOR=2',  # Good for categorical data
        'TILED=YES',
        'NUM_THREADS=ALL_CPUS'
    ]

    # Get dimensions
    H, W = Classified_Scene.shape

    # Create output dataset
    out_ds = driver.Create(
        output_path,
        W,
        H,
        1,  # Single band
        gdal.GDT_Byte,  # Unsigned 8-bit
        options=options
    )

    # Set georeferencing
    out_ds.SetGeoTransform(gdal_transform)
    out_ds.SetProjection(projection_wkt)

    # Write data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(Classified_Scene.astype(np.uint8))
    out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    out_band.SetNoDataValue(0)
    out_band.FlushCache()
    out_ds = None


def generate_coordinate_grid_256x256(geotransform, projection=None, patch_size=256):
    """
    Generate coordinate grids for a 256x256 patch using GDAL geotransform
    """
    if geotransform is None:
        return (np.zeros((patch_size, patch_size), dtype=np.float32),
                np.zeros((patch_size, patch_size), dtype=np.float32))

    # Create coordinate arrays using vectorized operations
    cols, rows = np.meshgrid(np.arange(patch_size), np.arange(patch_size))

    # GDAL geotransform formula:
    # X = gt[0] + col * gt[1] + row * gt[2]
    # Y = gt[3] + col * gt[4] + row * gt[5]
    x_coords = geotransform[0] + cols * geotransform[1] + rows * geotransform[2]
    y_coords = geotransform[3] + cols * geotransform[4] + rows * geotransform[5]

    # Convert to lat/lon if projected coordinates
    if projection and 'PROJCS' in projection:
        try:
            source_srs = osr.SpatialReference()
            source_srs.ImportFromWkt(projection)
            target_srs = osr.SpatialReference()
            target_srs.ImportFromEPSG(4326)  # WGS84

            transform = osr.CoordinateTransformation(source_srs, target_srs)

            # Transform coordinates - flatten arrays for batch processing
            x_flat = x_coords.flatten()
            y_flat = y_coords.flatten()
            coords = np.column_stack((x_flat, y_flat, np.zeros_like(x_flat)))

            # Transform all points at once
            transformed_coords = np.array(transform.TransformPoints(coords))

            # Reshape back to 2D arrays
            lon_array = transformed_coords[:, 0].reshape(patch_size, patch_size).astype(np.float32)
            lat_array = transformed_coords[:, 1].reshape(patch_size, patch_size).astype(np.float32)

            return lon_array, lat_array

        except Exception as e:
            print(f"Coordinate transformation failed: {e}")
            # Fall back to original coordinates
            return x_coords.astype(np.float32), y_coords.astype(np.float32)
    else:
        # Assume already in geographic coordinates
        return x_coords.astype(np.float32), y_coords.astype(np.float32)


def patch_generator_gdal(scene_dir, row_start, W, chunk_height, patch_size=256):
    """
    Generator that yields 256x256 patches with correct coordinates using GDAL
    """
    # Open the dataset with GDAL
    ds = gdal.Open(scene_dir)
    if ds is None:
        raise ValueError(f"Could not open file: {scene_dir}")

    # Get geotransform and projection
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    # Read the entire window
    merged_chunk = ds.ReadAsArray(0, row_start, W, chunk_height)
    print("unqiue", np.unique(merged_chunk))
    print(f"{row_start}-{W}-{chunk_height}")
    merged_chunk = merged_chunk.transpose(1, 2, 0)  # (H, W, C)


    H, W, C = merged_chunk.shape

    # Process in 256x256 patches
    for r in range(0, H, patch_size):
        for c in range(0, W, patch_size):
            # Extract image patch
            actual_height = min(patch_size, H - r)
            actual_width = min(patch_size, W - c)

            patch = merged_chunk[
                    r:r + actual_height,
                    c:c + actual_width,
                    :
                    ]
            hh_patch, hv_patch = patch[:, :, 0], patch[:, :, 1]
            channels = []
            channels.append(hh_patch.astype(np.float32))
            channels.append(hv_patch.astype(np.float32))
            hh_hv_ratio = _calculate_hh_hv_ratio(hh_patch, hv_patch)
            channels.append(hh_hv_ratio * 255)
            multi_channel = np.stack(channels, axis=0)

            # Calculate the geotransform for this specific patch
            patch_geotransform = (
                geotransform[0] + c * geotransform[1] + (row_start + r) * geotransform[2],  # x origin
                geotransform[1],  # x pixel size
                geotransform[2],  # x rotation
                geotransform[3] + c * geotransform[4] + (row_start + r) * geotransform[5],  # y origin
                geotransform[4],  # y rotation
                geotransform[5]  # y pixel size
            )


            # Generate coordinates for this exact patch
            lon_patch, lat_patch = generate_coordinate_grid_256x256(
                patch_geotransform, projection, patch_size
            )

            # If the patch is smaller than 256x256 (edge case), crop the coordinates
            if actual_height < patch_size or actual_width < patch_size:
                lon_patch = lon_patch[:actual_height, :actual_width]
                lat_patch = lat_patch[:actual_height, :actual_width]

            yield r, c, multi_channel, lon_patch, lat_patch

    ds = None  # Close the dataset


def classify_scene_with_generator(
        scene_dir,
        row_start,
        W,
        chunk_height,
        trained_model,
        uncertainty_estimator,
        patch_size=256,
        num_classes=14,
        device=None,
        batch_size=32,
        max_patches=None
):
    """
    Modified to use GDAL-based patch generator
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()

    # Initialize output arrays
    classified_scene = np.zeros((chunk_height, W), dtype=np.uint8)
    uncertainty_scene = np.zeros((chunk_height, W), dtype=np.float32)

    patches = []
    coords = []
    longtitude_set = []
    latitude_set = []
    patch_counter = 0

    # Calculate total patches for progress bar
    H = chunk_height
    total_patches = (H // patch_size + (1 if H % patch_size > 0 else 0)) * (
                W // patch_size + (1 if W % patch_size > 0 else 0))
    if max_patches is not None:
        total_patches = min(total_patches, max_patches)

    # Use GDAL-based patch generator
    for r, c, patch, long_, lat_ in tqdm(patch_generator_gdal(scene_dir, row_start, W, chunk_height, patch_size),
                                         total=total_patches, desc="Classifying patches"):
        if max_patches is not None and patch_counter >= max_patches:
            break

        # Pad patch if at border
        patch = patch.transpose(1,2,0)
        ph, pw, _ = patch.shape
        if ph < patch_size or pw < patch_size:
            pad_h = patch_size - ph
            pad_w = patch_size - pw
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            long_ = np.pad(long_, ((0, pad_h), (0, pad_w)), mode='reflect')
            lat_ = np.pad(lat_, ((0, pad_h), (0, pad_w)), mode='reflect')

        patches.append(patch)
        longtitude_set.append(long_)
        latitude_set.append(lat_)
        coords.append((r, c, ph, pw))

        patch_counter += 1

        if len(patches) == batch_size:
            batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float().to(device)
            longtitude_ = torch.from_numpy(np.stack(longtitude_set)).float().to(device)
            latitude_ = torch.from_numpy(np.stack(latitude_set)).float().to(device)
            with torch.no_grad():
                logits = trained_model(batch, long=longtitude_, lat=latitude_)[0]
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                print("pred unique", np.unique(preds))

            for idx, (row, col, ph, pw) in enumerate(coords):
                classified_scene[row:row + ph, col:col + pw] = preds[idx][:ph, :pw]

            patches = []
            coords = []
            latitude_set = []
            longtitude_set = []

    # Handle remaining patches
    if patches:
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float().to(device)
        longtitude_ = torch.from_numpy(np.stack(longtitude_set)).float().to(device)
        latitude_ = torch.from_numpy(np.stack(latitude_set)).float().to(device)

        with torch.no_grad():

            logits = trained_model(batch, long=longtitude_, lat=latitude_)[0]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        for idx, (row, col, ph, pw) in enumerate(coords):
            classified_scene[row:row + ph, col:col + pw] = preds[idx][:ph, :pw]

    return classified_scene


# Add band-specific ranges for preprocessing
band_ranges = {
    0: (900, 3000),  # Band B02
    1: (1000, 3000),  # Band B03
    2: (1000, 3000),  # Band B04
    3: (800, 8000),  # Band B08
    4: (1000, 4000),  # Band B05
    5: (900, 6000),  # Band B06
    6: (900, 8000),  # Band B07
    7: (900, 8000),  # Band B8A
    8: (1000, 6000),  # Band B11
    9: (1000, 4000),  # Band B12
}


def _calculate_hh_hv_ratio(hh_patch, hv_patch):
    """Calculate HH/HV ratio with safety against division by zero and invalid values."""
    hh_safe = hh_patch.astype(np.float32)
    hv_safe = hv_patch.astype(np.float32)
    hv_safe[hv_safe < 1e-6] = 1e-6
    ratio = hh_safe / hv_safe
    ratio[~np.isfinite(ratio)] = 0.0
    return ratio.astype(np.float32)


def _normalize_band(band_data, band_idx):
    min_val, max_val = band_ranges[band_idx]
    clipped = np.clip(band_data, min_val, max_val)
    normalized = clipped / max_val
    return normalized


def _normalize_image(image):
    hh_patch, hv_patch = image[:, :, 0], image[:, :, 1]
    channels = []
    channels.append(hh_patch)
    channels.append(hv_patch)
    hh_hv_ratio = _calculate_hh_hv_ratio(hh_patch, hv_patch)
    channels.append(hh_hv_ratio * 255)
    multi_channel = np.stack(channels, axis=2)
    return multi_channel


def map_generation(device, model, uncertainty_estimator, num_classes, accuracy_filename, output_dir, output_map_name,
                   model_path, sentinel2_img, uncertanity_map_name, class_label_list):
    device = device
    num_classes = num_classes
    output_filename = accuracy_filename
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_map_name)
    uncertainty_map_path = os.path.join(output_dir, uncertanity_map_name)

    # Load model weights
    ckpt_path = model_path
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    scene_dir = sentinel2_img
    ref_ds = gdal.Open(scene_dir)
    if ref_ds is None:
        raise ValueError(f"Could not open reference image: {scene_dir}")

    # Get dimensions
    W = ref_ds.RasterXSize
    H_total = ref_ds.RasterYSize
    W = W // 2

    Htotal = H_total // 2
    gdal_transform = ref_ds.GetGeoTransform()
    projection_wkt = ref_ds.GetProjection()

    print(f"Full scene dimensions: H={H_total}, W={W}")

    # Prepare full output array
    Classified_Scene = np.zeros((H_total, W), dtype=np.uint8)
    Uncertainty_Scene = np.zeros((H_total, W), dtype=np.float32)

    # Process in chunks for memory efficiency
    chunk_size = 300
    num_chunks = (H_total + chunk_size - 1) // chunk_size

    total_start_time = time.time()

    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        chunk_start_time = time.time()
        row_start = chunk_idx * chunk_size
        row_end = min(row_start + chunk_size, H_total)
        chunk_height = row_end - row_start

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}: rows {row_start}-{row_end - 1}")

        # Read chunk using GDAL
        ds = gdal.Open(scene_dir)
        merged_chunk = ds.ReadAsArray(0, row_start, W, chunk_height)
        merged_chunk = merged_chunk.transpose(1, 2, 0)
        print("unique", np.unique(merged_chunk))
        ds = None

        # Preprocess
        merged_chunk = _normalize_image(merged_chunk)
        print(f" Chunk shape after normalization: {merged_chunk.shape}")

        # Classify this chunk using GDAL-based generator
        try:
            classified_chunk = classify_scene_with_generator(
                scene_dir=scene_dir,
                row_start=row_start,
                W=W,
                chunk_height=chunk_height,
                trained_model=model,
                uncertainty_estimator=uncertainty_estimator,
                patch_size=256,
                num_classes=num_classes,
                batch_size=32,
                max_patches=None
            )

            # Save chunk results to full output array
            Classified_Scene[row_start:row_end, :] = classified_chunk
            chunk_time = time.time() - chunk_start_time
            print(f"Chunk {chunk_idx + 1} completed in {chunk_time:.2f} seconds")

        except Exception as e:
            print(f"[ERROR] Failed to classify chunk {chunk_idx}: {e}")
            traceback.print_exc()
            continue

        del merged_chunk, classified_chunk

    total_time = (time.time() - total_start_time) / 60
    print(f"\n[INFO] Total classification completed in {total_time:.2f} minutes")

    # Save results
    np.save(output_path.replace('.tif', '.npy'), Classified_Scene)
    print(f"Saving to: {output_path}")
    print(f"Classified_Scene shape: {Classified_Scene.shape}")

    # Save as GeoTIFF
    write_classified_geotiff_with_colormap(output_path, Classified_Scene, Uncertainty_Scene, uncertainty_map_path,
                                           gdal_transform, projection_wkt)
    print(f"[INFO] Final classified map saved with geospatial info: {output_path}")