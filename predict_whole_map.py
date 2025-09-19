import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal, osr
import rasterio
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class PredictionDataset(Dataset):
    def __init__(self, data_dir, transform=None, add_hh_hv_ratio=True, ratio_epsilon=1e-6):
        """
        Dataset for prediction with longitude and latitude arrays
        For HH/HV pairs with naming pattern: base_name_HH.tiff + base_name_HV.tiff
        """
        self.data_dir = data_dir
        self.transform = transform
        self.add_hh_hv_ratio = add_hh_hv_ratio
        self.ratio_epsilon = ratio_epsilon

        # Find all HH/HV image pairs and precompute geospatial info
        self.file_pairs = self._find_hh_hv_pairs()
        self.geo_info = self._precompute_geospatial_info()

        print(f"Found {len(self.file_pairs)} HH/HV image pairs")

    def _find_hh_hv_pairs(self):
        """Find HH/HV image pairs with pattern: base_name_HH.tiff + base_name_HV.tiff"""
        file_pairs = []

        # Find all HH files
        hh_files = glob.glob(os.path.join(self.data_dir, "*_HH.tiff"))

        for hh_file in hh_files:
            # Construct corresponding HV file path
            hv_file = hh_file.replace("_HH.tiff", "_HV.tiff")

            if os.path.exists(hv_file):
                # Extract base name (e.g., "patch_10086_10086")
                base_name = os.path.basename(hh_file).replace("_HH.tiff", "")
                file_pairs.append((hh_file, hv_file, base_name))

        return file_pairs

    def _precompute_geospatial_info(self):
        """Precompute all geospatial information to avoid complex objects in __getitem__"""
        geo_info = []

        for hh_path, hv_path, base_name in self.file_pairs:
            try:
                ds = gdal.Open(hh_path)
                if ds is None:
                    geo_info.append({
                        'geotransform': None,
                        'projection_wkt': None,
                        'crs_epsg': None,
                        'filename': os.path.basename(hh_path),
                        'base_name': base_name
                    })
                    continue

                geotransform = ds.GetGeoTransform()
                projection = ds.GetProjection()

                # Convert CRS to EPSG code if possible
                crs_epsg = None
                if projection:
                    try:
                        srs = osr.SpatialReference()
                        srs.ImportFromWkt(projection)
                        crs_epsg = srs.GetAuthorityCode(None)
                    except:
                        crs_epsg = None

                ds = None

                geo_info.append({
                    'geotransform': geotransform,
                    'projection_wkt': projection,
                    'crs_epsg': crs_epsg,
                    'filename': os.path.basename(hh_path),
                    'base_name': base_name
                })

            except Exception as e:
                print(f"Error getting geospatial info for {base_name}: {e}")
                geo_info.append({
                    'geotransform': None,
                    'projection_wkt': None,
                    'crs_epsg': None,
                    'filename': os.path.basename(hh_path),
                    'base_name': base_name
                })

        return geo_info

    def __len__(self):
        return len(self.file_pairs)

    def _read_geotiff(self, file_path):
        """Read a GeoTIFF file and return as numpy array."""
        ds = gdal.Open(file_path)
        if ds is None:
            raise ValueError(f"Could not open file: {file_path}")
        array = ds.ReadAsArray()
        ds = None
        return array

    def _create_lon_lat_arrays(self, geotransform, projection_wkt, height, width):
        """Create arrays of longitude and latitude coordinates."""
        if geotransform is None:
            return (np.zeros((height, width), dtype=np.float32),
                    np.zeros((height, width), dtype=np.float32))

        x_coords = np.zeros((height, width), dtype=np.float64)
        y_coords = np.zeros((height, width), dtype=np.float64)

        for row in range(height):
            for col in range(width):
                x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
                y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
                x_coords[row, col] = x
                y_coords[row, col] = y

        if projection_wkt and 'PROJCS' in projection_wkt:
            try:
                source_srs = osr.SpatialReference()
                source_srs.ImportFromWkt(projection_wkt)
                target_srs = osr.SpatialReference()
                target_srs.ImportFromEPSG(4326)

                transform = osr.CoordinateTransformation(source_srs, target_srs)
                x_flat = x_coords.flatten()
                y_flat = y_coords.flatten()
                coords = np.column_stack((x_flat, y_flat, np.zeros_like(x_flat)))
                transformed_coords = np.array(transform.TransformPoints(coords))

                lon_array = transformed_coords[:, 0].reshape(height, width).astype(np.float32)
                lat_array = transformed_coords[:, 1].reshape(height, width).astype(np.float32)

                return lon_array, lat_array
            except Exception as e:
                print(f"Coordinate transformation failed: {e}")
                return x_coords.astype(np.float32), y_coords.astype(np.float32)
        else:
            return x_coords.astype(np.float32), y_coords.astype(np.float32)

    def _calculate_hh_hv_ratio(self, hh_patch, hv_patch):
        """Calculate HH/HV ratio with safety against division by zero."""
        hh_safe = hh_patch.astype(np.float32)
        hv_safe = hv_patch.astype(np.float32)
        hv_safe[hv_safe < self.ratio_epsilon] = self.ratio_epsilon
        ratio = hh_safe / hv_safe
        ratio[~np.isfinite(ratio)] = 0.0
        return ratio.astype(np.float32)

    def _create_multi_channel_input(self, hh_patch, hv_patch):
        """Create multi-channel input from HH, HV, and optionally HH/HV ratio."""
        channels = []
        channels.append(hh_patch.astype(np.float32))
        channels.append(hv_patch.astype(np.float32))

        if self.add_hh_hv_ratio:
            hh_hv_ratio = self._calculate_hh_hv_ratio(hh_patch, hv_patch)
            channels.append(hh_hv_ratio)

        return np.stack(channels, axis=0)

    def __getitem__(self, idx):
        hh_path, hv_path, base_name = self.file_pairs[idx]
        geo_info = self.geo_info[idx]

        try:
            # Read images
            hh_patch = self._read_geotiff(hh_path)
            hv_patch = self._read_geotiff(hv_path)

            # Get actual dimensions
            height, width = hh_patch.shape

            # Create longitude and latitude arrays
            lon_array, lat_array = self._create_lon_lat_arrays(
                geo_info['geotransform'],
                geo_info['projection_wkt'],
                height, width
            )

            # Create input stack
            input_data = self._create_multi_channel_input(hh_patch, hv_patch)

            # Convert to tensors
            input_tensor = torch.from_numpy(input_data)
            lon_tensor = torch.from_numpy(lon_array.astype(np.float32))
            lat_tensor = torch.from_numpy(lat_array.astype(np.float32))

            if self.transform:
                input_tensor = self.transform(input_tensor)

            # Return only simple data types that can be collated
            return {
                'input': input_tensor,
                'longitude': lon_tensor,
                'latitude': lat_tensor,
                'geotransform': geo_info['geotransform'],  # tuple
                'projection_wkt': geo_info['projection_wkt'],  # string
                'crs_epsg': geo_info['crs_epsg'],  # string or None
                'filename': geo_info['filename'],  # string
                'base_name': geo_info['base_name'],  # string
                'num_channels': input_data.shape[0],  # int
                'height': height,  # int
                'width': width  # int
            }

        except Exception as e:
            print(f"Error loading {base_name}: {e}")
            # Return a different sample if this one fails
            return self.__getitem__((idx + 1) % len(self))


# Custom collate function to handle mixed data types
def custom_collate_fn1(batch):
    """Custom collate function to handle mixed tensor and non-tensor data"""
    elem = batch[0]
    collated = {}

    for key in elem.keys():
        if isinstance(elem[key], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(elem[key], (int, float, str)) or elem[key] is None:
            # Store as list
            collated[key] = [item[key] for item in batch]
        elif isinstance(elem[key], tuple):
            # Store tuples as list
            collated[key] = [item[key] for item in batch]
        else:
            # Convert other types to string or handle appropriately
            collated[key] = [str(item[key]) if item[key] is not None else None for item in batch]

    return collated


class PredictionHandler:
    def __init__(self, output_dir, model, class_names=None):
        """
        Handles prediction and saving of results with geospatial information
        """
        self.output_dir = output_dir
        self.model = model
        self.model.eval()
        self.class_names = class_names or {
            0: 'first_year_ice',
            1: 'old_ice',
            2: 'water'
        }
        os.makedirs(output_dir, exist_ok=True)

    def predict_and_save(self, dataloader, device='cuda'):
        """
        Run prediction on dataloader and save results with geospatial info
        """
        self.model.to(device)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs = batch['input'].to(device)
                long = batch['longitude'].to(device)
                lat = batch['latitude'].to(device)

                # Get predictions
                outputs = self.model(inputs, long=long, lat=lat)[0]
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                # Save each prediction with geospatial info
                for i in range(len(preds)):
                    self._save_prediction(
                        preds[i],
                        batch['geotransform'][i],
                        batch['projection_wkt'][i],
                        batch['base_name'][i],
                        batch['longitude'][i].numpy(),
                        batch['latitude'][i].numpy(),
                        batch['height'][i],
                        batch['width'][i]
                    )

    def _save_prediction(self, prediction, geotransform, projection_wkt, base_name,
                         lon_array, lat_array, height, width):
        """
        Save individual prediction with geospatial information
        """
        output_path = os.path.join(self.output_dir, f"{base_name}_prediction.tiff")

        # Create metadata for GeoTIFF
        metadata = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8',
        }

        # Add CRS if available
        if projection_wkt:
            metadata['crs'] = projection_wkt

        # Add transform if available
        if geotransform and isinstance(geotransform, (tuple, list)) and len(geotransform) == 6:
            metadata['transform'] = rasterio.Affine.from_gdal(*geotransform)

        # Save prediction with geospatial info
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(prediction.astype(np.uint8), 1)

        # Save coordinate arrays
        self._save_coordinate_arrays(base_name, lon_array, lat_array)

        print(f"Saved prediction: {output_path}")

    def _save_coordinate_arrays(self, base_name, lon_array, lat_array):
        """Save longitude and latitude arrays as numpy files."""
        np.save(os.path.join(self.output_dir, f"{base_name}_longitude.npy"), lon_array)
        np.save(os.path.join(self.output_dir, f"{base_name}_latitude.npy"), lat_array)

    def create_mosaic(self, output_mosaic_path):
        """
        Create mosaic from all prediction files
        """
        prediction_files = glob.glob(os.path.join(self.output_dir, "*_prediction.tiff"))

        if not prediction_files:
            raise ValueError("No prediction files found to create mosaic")

        print(f"Creating mosaic from {len(prediction_files)} prediction files...")

        vrt_path = os.path.join(self.output_dir, "predictions.vrt")

        # Build VRT from all prediction files
        vrt = gdal.BuildVRT(vrt_path, prediction_files)
        if vrt is None:
            raise ValueError("Failed to build VRT file")
        vrt = None

        # Translate VRT to final mosaic
        mosaic = gdal.Translate(output_mosaic_path, vrt_path, format='GTiff')
        if mosaic is None:
            raise ValueError("Failed to create mosaic")
        mosaic = None

        # Clean up temporary VRT
        if os.path.exists(vrt_path):
            os.remove(vrt_path)

        print(f"Mosaic created: {output_mosaic_path}")


# Example usage
def main():
    # Configuration
    data_dir = "/path/to/your/images"
    output_dir = "/path/to/output/predictions"
    mosaic_path = "/path/to/output/mosaic.tiff"

    # Create dataset and dataloader for prediction with custom collate
    dataset = PredictionDataset(data_dir, add_hh_hv_ratio=True)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn1
    )

    # Load your trained model
    class DummyModel(torch.nn.Module):
        def __init__(self, in_channels=3):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

        def forward(self, x):
            return torch.nn.functional.log_softmax(self.conv(x), dim=1)

    # Check number of channels from a sample
    sample = dataset[0]
    in_channels = sample['num_channels']
    print(f"Input channels: {in_channels}")

    model = DummyModel(in_channels=in_channels)

    # Initialize prediction handler
    predictor = PredictionHandler(output_dir, model)

    # Run prediction and save results
    predictor.predict_and_save(dataloader, device='cpu')

    # Create mosaic from all predictions
    predictor.create_mosaic(mosaic_path)

    print("Prediction and mosaic creation completed!")


if __name__ == "__main__":
    main()