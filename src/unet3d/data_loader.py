import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import segyio


class SeismicDataset(Dataset):
    def __init__(self, seismic_dir, fault_dir, dim, use_osv=False):
        """
        Args:
            seismic_path: Directory containing seismic data files
            fault_path: Directory containing fault data files
            filenames: List of filenames (without path)
            dim: Tuple of dimensions for reshaping
        """
        self.seismic_dir = seismic_dir
        self.fault_dir = fault_dir
        self.filenames = os.listdir(fault_dir)
        self.dim = dim
        self.use_osv = use_osv

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # Load seismic and fault data using full paths
        seismic = np.fromfile(os.path.join(self.seismic_dir, filename), dtype=np.single)
        fault = np.fromfile(os.path.join(self.fault_dir, filename), dtype=np.single)

        # Reshape
        seismic = np.reshape(seismic, self.dim)
        fault = np.reshape(fault, self.dim)

        if not self.use_osv:
            # Normalize seismic data
            xm = np.mean(seismic)
            xs = np.std(seismic)
            seismic = (seismic - xm) / xs
            seismic = np.transpose(seismic)

            fault = np.transpose(fault)

        seismic = np.expand_dims(seismic, 0)
        fault = np.expand_dims(fault, 0)

        # Convert to torch tensors
        return torch.FloatTensor(seismic), torch.FloatTensor(fault)


class SegyData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.inline_numbers = np.array([])
        self.crossline_numbers = np.array([])
        self.seismic_array = None  # Replace dict with NumPy array

    def load_segy(self):
        """Load SEGY data and store it in a NumPy array instead of a dictionary."""
        with segyio.open(self.file_path, "r", ignore_geometry=True) as segyfile:
            segyfile.mmap()

            num_traces = segyfile.tracecount
            inline_set = set()
            crossline_set = set()
            trace_list = []

            for trace_index in range(num_traces):
                header = segyfile.header[trace_index]
                inline = header.get(segyio.TraceField.INLINE_3D, None)
                crossline = header.get(segyio.TraceField.CROSSLINE_3D, None)

                if inline is not None and crossline is not None:
                    inline_set.add(inline)
                    crossline_set.add(crossline)
                    trace_list.append((inline, crossline, segyfile.trace[trace_index]))

            # Convert to sorted NumPy arrays
            self.inline_numbers = np.sort(np.array(list(inline_set)))
            self.crossline_numbers = np.sort(np.array(list(crossline_set)))

            # Get dimensions
            num_inlines = len(self.inline_numbers)
            num_crosslines = len(self.crossline_numbers)
            num_samples = len(trace_list[0][2])  # Get trace length from first trace

            # Create a 3D NumPy array filled with NaNs
            self.seismic_array = np.full((num_inlines, num_crosslines, num_samples), np.nan, dtype=np.float32)

            # Map inline/crossline to array indices
            inline_idx_map = {v: i for i, v in enumerate(self.inline_numbers)}
            crossline_idx_map = {v: i for i, v in enumerate(self.crossline_numbers)}

            # Fill the array
            for inline, crossline, trace in trace_list:
                i = inline_idx_map[inline]
                j = crossline_idx_map[crossline]
                self.seismic_array[i, j, :] = trace  # Assign trace data

            print(f"Loaded {num_traces} traces into a {num_inlines}x{num_crosslines}x{num_samples} NumPy array.")

    def get_available_inlines_crosslines(self):
        """Returns inline and crossline numbers."""
        return self.inline_numbers, self.crossline_numbers

    def plot_trace_coverage(self):
        """Plot inline vs crossline trace coverage."""
        if self.seismic_array is None:
            print("No data available. Please load the SEGY file first.")
            return

        # Find available traces (non-NaN values)
        inlines, crosslines = np.where(~np.isnan(self.seismic_array[:, :, 0]))

        plt.figure(figsize=(10, 6))
        plt.scatter(self.crossline_numbers[crosslines], self.inline_numbers[inlines], s=1, color="blue", alpha=0.5)
        plt.xlabel("Crossline")
        plt.ylabel("Inline")
        plt.title("Seismic Trace Coverage")
        plt.grid(True)
        plt.show()

    def plot_seismic_section(self, section_type="inline", section_value=None):
        """Plot a seismic section for a specific inline or crossline."""
        if self.seismic_array is None:
            print("No seismic data available. Please load the SEGY file first.")
            return

        if section_type == "inline":
            if section_value not in self.inline_numbers:
                print(f"Inline {section_value} not found.")
                return
            index = np.where(self.inline_numbers == section_value)[0][0]
            section_data = self.seismic_array[index, :, :]
        elif section_type == "crossline":
            if section_value not in self.crossline_numbers:
                print(f"Crossline {section_value} not found.")
                return
            index = np.where(self.crossline_numbers == section_value)[0][0]
            section_data = self.seismic_array[:, index, :]
        else:
            print("Invalid section type. Choose 'inline' or 'crossline'.")
            return

        if np.isnan(section_data).all():
            print(f"No data available for {section_type} {section_value}.")
            return

        plt.figure(figsize=(10, 6))
        plt.imshow(
            section_data.T,
            aspect="auto",
            cmap="seismic",
            interpolation="bilinear",
            vmin=-np.nanmax(np.abs(section_data)),
            vmax=np.nanmax(np.abs(section_data)),
        )
        plt.colorbar(label="Amplitude")
        plt.xlabel("Trace Index")
        plt.ylabel("Time Samples")
        plt.title(f"Seismic Section - {section_type.capitalize()} {section_value}")
        plt.show()


class SeismicData:
    def __init__(self, use_osv=False):
        self.use_osv = use_osv

    def load_dat(self, seismic_path, fault_path=None, dim=(128, 128, 128)):
        n1, n2, n3 = dim
        # Load and preprocess data
        seismic = np.fromfile(seismic_path, dtype=np.single)
        seismic = np.reshape(seismic, (n1, n2, n3))

        if not self.use_osv:
            # Normalize
            gm = np.mean(seismic)
            gs = np.std(seismic)
            seismic = (seismic - gm) / gs
            seismic = np.transpose(seismic)

        seismic = np.expand_dims(seismic, axis=(0, 1))  # Add batch and channel dimensions

        if fault_path is not None:
            fault = np.fromfile(fault_path, dtype=np.single)
            fault = np.reshape(fault, (n1, n2, n3))
            if not self.use_osv:
                fault = np.transpose(fault)
            fault = np.expand_dims(fault, axis=(0, 1))
            return seismic, fault
        else:
            return seismic

    def load_segy(self, segy_path, patch_size=(128, 128, 128)):
        # Instantiate and load data using SegyData
        segy_data_loader = SegyData(segy_path)
        segy_data_loader.load_segy()
        # from (inlines, xlines, samples) -> (samples, inlines, xlines)
        seismic_array = np.transpose(segy_data_loader.seismic_array, (2, 0, 1)).astype(np.float32)

        p0, p1, p2 = patch_size
        n0, n1, n2 = seismic_array.shape
        patches = []

        # Slice into non-overlapping patches
        for i in range(0, n0, p0):
            if i + p0 > n0:
                continue
            for j in range(0, n1, p1):
                if j + p1 > n1:
                    continue
                for k in range(0, n2, p2):
                    if k + p2 > n2:
                        continue
                    patch = seismic_array[i : i + p0, j : j + p1, k : k + p2]
                    if not self.use_osv:
                        # Normalize the patch
                        m = np.mean(patch)
                        s = np.std(patch)
                        patch = (patch - m) / s if s > 0 else patch
                        # Transpose: data shape becomes (samples, inlines, crosslines)
                        patch = np.transpose(patch)
                    # Add batch and channel dimensions
                    patch = np.expand_dims(patch, axis=(0, 1))
                    patches.append(patch)
        print(f"Extracted {len(patches)} patches from the SEGY data.")
        return patches

    def load_dataset(self, seismic_path, fault_path, dim=(128, 128, 128), batch_size=4, num_workers=1):
        dataset = SeismicDataset(seismic_path, fault_path, dim, self.use_osv)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return data_loader
