import numpy as np
from osgeo import gdal


def process_mosaic(mosaic_path, nodata_mask_path, land_mask_path, output_path):
    # Open all files
    mosaic_ds = gdal.Open(mosaic_path, gdal.GA_ReadOnly)
    nodata_ds = gdal.Open(nodata_mask_path, gdal.GA_ReadOnly)
    land_ds = gdal.Open(land_mask_path, gdal.GA_ReadOnly)

    # Get mosaic geotransform and dimensions
    mosaic_gt = mosaic_ds.GetGeoTransform()
    mosaic_xsize = mosaic_ds.RasterXSize
    mosaic_ysize = mosaic_ds.RasterYSize

    # Read the entire mosaic array and apply value transformation
    mosaic_array = mosaic_ds.GetRasterBand(1).ReadAsArray()
    output_array = mosaic_array + 1  # Apply value transformation: 0→1, 1→2, 2→3

    # Calculate the geographic bounds of the mosaic
    mosaic_ulx = mosaic_gt[0]
    mosaic_uly = mosaic_gt[3]
    mosaic_lrx = mosaic_gt[0] + mosaic_gt[1] * mosaic_xsize
    mosaic_lry = mosaic_gt[3] + mosaic_gt[5] * mosaic_ysize

    # Process both masks
    masks = [
        ('nodata', nodata_ds),
        ('land', land_ds)
    ]

    for mask_name, mask_ds in masks:
        mask_gt = mask_ds.GetGeoTransform()
        mask_xsize = mask_ds.RasterXSize
        mask_ysize = mask_ds.RasterYSize

        # Calculate the geographic bounds of the mask
        mask_ulx = mask_gt[0]
        mask_uly = mask_gt[3]
        mask_lrx = mask_gt[0] + mask_gt[1] * mask_xsize
        mask_lry = mask_gt[3] + mask_gt[5] * mask_ysize

        # Find the overlapping bounding box
        overlap_ulx = max(mosaic_ulx, mask_ulx)
        overlap_uly = min(mosaic_uly, mask_uly)
        overlap_lrx = min(mosaic_lrx, mask_lrx)
        overlap_lry = max(mosaic_lry, mask_lry)

        # Check if there's any overlap
        if overlap_ulx >= overlap_lrx or overlap_uly <= overlap_lry:
            print(f"Warning: No overlap between mosaic and {mask_name} mask")
            continue

        # Convert geographic coordinates to pixel coordinates for mosaic
        mosaic_xoff = int((overlap_ulx - mosaic_gt[0]) / mosaic_gt[1] + 0.5)
        mosaic_yoff = int((overlap_uly - mosaic_gt[3]) / mosaic_gt[5] + 0.5)
        mosaic_xsize_overlap = int((overlap_lrx - overlap_ulx) / mosaic_gt[1] + 0.5)
        mosaic_ysize_overlap = int((overlap_lry - overlap_uly) / mosaic_gt[5] + 0.5)

        # Ensure the overlap coordinates are within mosaic bounds
        mosaic_xoff = max(0, min(mosaic_xoff, mosaic_xsize - 1))
        mosaic_yoff = max(0, min(mosaic_yoff, mosaic_ysize - 1))
        mosaic_xsize_overlap = min(mosaic_xsize_overlap, mosaic_xsize - mosaic_xoff)
        mosaic_ysize_overlap = min(mosaic_ysize_overlap, mosaic_ysize - mosaic_yoff)

        # Convert geographic coordinates to pixel coordinates for mask
        mask_xoff = int((overlap_ulx - mask_gt[0]) / mask_gt[1] + 0.5)
        mask_yoff = int((overlap_uly - mask_gt[3]) / mask_gt[5] + 0.5)
        mask_xsize_overlap = int((overlap_lrx - overlap_ulx) / mask_gt[1] + 0.5)
        mask_ysize_overlap = int((overlap_lry - overlap_uly) / mask_gt[5] + 0.5)

        # Ensure the overlap coordinates are within mask bounds
        mask_xoff = max(0, min(mask_xoff, mask_xsize - 1))
        mask_yoff = max(0, min(mask_yoff, mask_ysize - 1))
        mask_xsize_overlap = min(mask_xsize_overlap, mask_xsize - mask_xoff)
        mask_ysize_overlap = min(mask_ysize_overlap, mask_ysize - mask_yoff)

        # Use the smaller of the two overlap dimensions
        final_xsize_overlap = min(mosaic_xsize_overlap, mask_xsize_overlap)
        final_ysize_overlap = min(mosaic_ysize_overlap, mask_ysize_overlap)

        if final_xsize_overlap <= 0 or final_ysize_overlap <= 0:
            print(f"Warning: No valid overlap area for {mask_name} mask")
            continue

        # Read the overlapping areas with the same dimensions
        mosaic_overlap = mosaic_ds.GetRasterBand(1).ReadAsArray(
            mosaic_xoff, mosaic_yoff, final_xsize_overlap, final_ysize_overlap
        )

        mask_overlap = mask_ds.GetRasterBand(1).ReadAsArray(
            mask_xoff, mask_yoff, final_xsize_overlap, final_ysize_overlap
        )

        # Apply masking based on mask type
        if mask_name == 'nodata':
            # For no-data mask: value 1 = no-data area → set to 0
            masked_overlap = np.where(mask_overlap == 1, 0, mosaic_overlap + 1)
        elif mask_name == 'land':
            # For land mask: value 1 = land area → set to 0
            masked_overlap = np.where(mask_overlap == 1, 0, mosaic_overlap + 1)

        # Update the output array with the masked overlapping area
        output_array[mosaic_yoff:mosaic_yoff + final_ysize_overlap,
        mosaic_xoff:mosaic_xoff + final_xsize_overlap] = masked_overlap

        print(f"{mask_name.capitalize()} mask applied to area: {final_xsize_overlap}x{final_ysize_overlap} pixels")

    # Create and save the output
    create_output(output_array, mosaic_ds, output_path)

    # Close datasets
    mosaic_ds = None
    nodata_ds = None
    land_ds = None

    print(f"Processing complete. Output saved to: {output_path}")


def create_output(array, reference_ds, output_path):
    """Create output GeoTIFF file"""
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path,
                           array.shape[1],
                           array.shape[0],
                           1,
                           gdal.GDT_Byte)

    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)

    # Optional: set no-data value
    out_band.SetNoDataValue(0)

    out_ds = None


# Usage example
if __name__ == "__main__":
    mosaic_file = "/mnt/storage/Yimin/Seaice/results/mosaic.tiff"
    nodata_mask_file = "/beluga/Hack12_multi_type_seaice/Hackathon_Sea_Ice_Typing/patch_extraction_script/nodata_mask.tif"  # 1 = no-data area
    land_mask_file = "/mnt/storage/Yimin/Seaice/land_mask.tif"  # 1 = land area
    output_file = "output_mosaic2.tif"

    process_mosaic(mosaic_file, nodata_mask_file, land_mask_file, output_file)