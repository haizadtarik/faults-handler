import torch
import torch.nn as nn


class UNet3D_OSV(nn.Module):
    """
    A 3D U-Net with a variable number of down/upsampling 'depth' levels.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for seismic).
        out_channels (int): Number of output channels (e.g., 1 for faults).
        base_channels (int): Number of filters in the first encoder block.
        depth (int): Number of times we downsample (and thus also upsample).

    Example:
        model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=4)
        # Depth=4 -> 4 downsamplings + bottom block + 4 upsamplings
    """

    def __init__(self, depth, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        self.depth = depth

        # ---------------------------------------------------
        # 1) Create a list of channel sizes for each level
        # ---------------------------------------------------
        # Example: if depth=4 and base_channels=16,
        # filters = [16, 32, 64, 128, 256]
        # The i-th encoder block produces filters[i] channels.
        # The bottleneck also produces filters[depth] channels.
        self.filters = [base_channels * (2**i) for i in range(depth + 1)]

        # ---------------------------------------------------
        # 2) Encoder: depth blocks + MaxPool after each (except last)
        # ---------------------------------------------------
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i in range(depth):
            out_ch = self.filters[i]
            block = self._make_conv_block(in_ch, out_ch)
            self.encoder_blocks.append(block)
            in_ch = out_ch
            self.pools.append(nn.MaxPool3d(kernel_size=2))

        # ---------------------------------------------------
        # 3) Bottleneck (the bottom block)
        # ---------------------------------------------------
        bottleneck_channels = self.filters[depth]
        self.bottleneck = self._make_conv_block(in_ch, bottleneck_channels)

        # ---------------------------------------------------
        # 4) Decoder: for each level, we do:
        #    - Transposed conv to upsample
        #    - Concat with skip from encoder
        #    - Conv block
        # ---------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Going "back up" from bottom => depth-1 ... 0
        for i in range(depth):
            # up_in = filters[depth - i]
            # up_out = filters[depth - 1 - i]
            up_in_channels = self.filters[depth - i]  # e.g. 256 -> 128 -> ...
            up_out_channels = self.filters[depth - 1 - i]  # e.g. 128 -> 64 -> ...

            # 4a) Transposed Conv
            up_block = nn.ConvTranspose3d(
                in_channels=up_in_channels, out_channels=up_out_channels, kernel_size=2, stride=2
            )
            self.up_blocks.append(up_block)

            # 4b) After upsampling, we will concat with an encoder skip that has
            #     skip_channels = filters[depth-1 - i].
            # The total channels going into the decoder conv block is
            #     up_out_channels + skip_channels
            # but skip_channels == up_out_channels by definition. So total_in = 2 * up_out_channels.
            decoder_in = up_out_channels * 2
            decoder_out = up_out_channels

            dec_block = self._make_conv_block(decoder_in, decoder_out)
            self.decoder_blocks.append(dec_block)

        # ---------------------------------------------------
        # 5) Final 1×1×1 Convolution
        # ---------------------------------------------------
        self.final_conv = nn.Conv3d(self.filters[0], out_channels, kernel_size=1)

    def _make_conv_block(self, in_ch, out_ch):
        """
        Helper: Two sequential 3D convolutions + ReLU.
        """
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass:
          - encode down, store skip connections
          - bottleneck
          - decode up, concat skip
          - final conv
        """
        # ------------------
        # Encoder
        # ------------------
        skips = []
        out = x
        for i in range(self.depth):
            out = self.encoder_blocks[i](out)  # conv block
            skips.append(out)
            out = self.pools[i](out)  # downsample

        # ------------------
        # Bottleneck
        # ------------------
        out = self.bottleneck(out)

        # ------------------
        # Decoder
        # ------------------
        for i in range(self.depth):
            # 1) upsample
            out = self.up_blocks[i](out)
            # 2) concat skip
            skip = skips[self.depth - 1 - i]
            out = torch.cat([out, skip], dim=1)
            # 3) decoder conv
            out = self.decoder_blocks[i](out)

        # ------------------
        # Final 1×1×1 conv
        # ------------------
        out = self.final_conv(out)
        return out


class UNet3D(nn.Module):
    """
    A 3D U-Net with a variable number of down/upsampling 'depth' levels.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for seismic).
        out_channels (int): Number of output channels (e.g., 1 for faults).
        base_channels (int): Number of filters in the first encoder block.
        depth (int): Number of times we downsample (and thus also upsample).

    Example:
        model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=4)
        # Depth=4 -> 4 downsamplings + bottom block + 4 upsamplings
    """

    def __init__(self, depth, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        self.depth = depth

        # ---------------------------------------------------
        # 1) Create a list of channel sizes for each level
        # ---------------------------------------------------
        # Example: if depth=4 and base_channels=16,
        # filters = [16, 32, 64, 128, 256]
        # The i-th encoder block produces filters[i] channels.
        # The bottleneck also produces filters[depth] channels.
        self.filters = [base_channels * (2**i) for i in range(depth + 1)]

        # ---------------------------------------------------
        # 2) Encoder: depth blocks + MaxPool after each (except last)
        # ---------------------------------------------------
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i in range(depth):
            out_ch = self.filters[i]
            block = self._make_conv_block(in_ch, out_ch)
            self.encoder_blocks.append(block)
            in_ch = out_ch
            self.pools.append(nn.MaxPool3d(kernel_size=2))

        # ---------------------------------------------------
        # 3) Bottleneck (the bottom block)
        # ---------------------------------------------------
        bottleneck_channels = self.filters[depth]
        self.bottleneck = self._make_conv_block(in_ch, bottleneck_channels)

        # ---------------------------------------------------
        # 4) Decoder: for each level, we do:
        #    - Transposed conv to upsample
        #    - Concat with skip from encoder
        #    - Conv block
        # ---------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Going "back up" from bottom => depth-1 ... 0
        for i in range(depth):
            # up_in = filters[depth - i]
            # up_out = filters[depth - 1 - i]
            up_in_channels = self.filters[depth - i]  # e.g. 256 -> 128 -> ...
            up_out_channels = self.filters[depth - 1 - i]  # e.g. 128 -> 64 -> ...

            # 4a) Transposed Conv
            up_block = nn.ConvTranspose3d(
                in_channels=up_in_channels, out_channels=up_out_channels, kernel_size=2, stride=2
            )
            self.up_blocks.append(up_block)

            # 4b) After upsampling, we will concat with an encoder skip that has
            #     skip_channels = filters[depth-1 - i].
            # The total channels going into the decoder conv block is
            #     up_out_channels + skip_channels
            # but skip_channels == up_out_channels by definition. So total_in = 2 * up_out_channels.
            decoder_in = up_out_channels * 2
            decoder_out = up_out_channels

            dec_block = self._make_conv_block(decoder_in, decoder_out)
            self.decoder_blocks.append(dec_block)

        # ---------------------------------------------------
        # 5) Final 1×1×1 Convolution
        # ---------------------------------------------------
        self.final_conv = nn.Conv3d(self.filters[0], out_channels, kernel_size=1)

    def _make_conv_block(self, in_ch, out_ch):
        """
        Helper: Two sequential 3D convolutions + ReLU.
        """
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass:
          - encode down, store skip connections
          - bottleneck
          - decode up, concat skip
          - final conv
        """
        # ------------------
        # Encoder
        # ------------------
        skips = []
        out = x
        for i in range(self.depth):
            out = self.encoder_blocks[i](out)  # conv block
            skips.append(out)
            out = self.pools[i](out)  # downsample

        # ------------------
        # Bottleneck
        # ------------------
        out = self.bottleneck(out)

        # ------------------
        # Decoder
        # ------------------
        for i in range(self.depth):
            # 1) upsample
            out = self.up_blocks[i](out)
            # 2) concat skip
            skip = skips[self.depth - 1 - i]
            out = torch.cat([out, skip], dim=1)
            # 3) decoder conv
            out = self.decoder_blocks[i](out)

        # ------------------
        # Final 1×1×1 conv
        # ------------------
        out = self.final_conv(out)
        return torch.sigmoid(out)
    