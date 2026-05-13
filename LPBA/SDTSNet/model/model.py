import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from swin_transformer import SwinTransformer


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(
        self,
        ndims,
        in_channels,
        out_channels,
        kernal_size=3,
        stride=1,
        padding=1,
        alpha=0.1,
    ):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class InsBlock(nn.Module):
    def __init__(self, channels, alpha=0.1):
        super().__init__()

        self.norm = nn.InstanceNorm3d(channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.norm(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(
        self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1
    ):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1
        )

        self.actout = InsBlock(out_channels, alpha)

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class ConvResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=0.1
    ):
        super(ConvResBlock, self).__init__()
        self.main = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.block = nn.Sequential(
            InsBlock(out_channels, alpha),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.actout = InsBlock(out_channels, alpha)

    def forward(self, x):
        x = self.main(x)
        out = self.block(x) + x
        return self.actout(out)


class ConvLSTM(nn.Module):
    def __init__(self, inputc, hidden_dim):
        super().__init__()
        self.convf = nn.Conv3d(inputc, hidden_dim, 3, 1, 1, bias=True)
        self.convi = nn.Conv3d(inputc, hidden_dim, 3, 1, 1, bias=True)
        self.convct = nn.Conv3d(inputc, hidden_dim, 3, 1, 1, bias=True)
        self.convo = nn.Conv3d(inputc, hidden_dim, 3, 1, 1, bias=True)

    def forward(self, C, h, x):
        hx = torch.cat([h, x], dim=1)
        f = torch.sigmoid(self.convf(hx))
        i = torch.sigmoid(self.convi(hx))
        Ct = torch.tanh(self.convct(hx))
        o = torch.sigmoid(self.convo(hx))
        C = f * C + i * Ct
        h = o * torch.tanh(C)
        return C, h


class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad3d(max_disp, 0)

    def forward(self, x_1, x_2):

        x_2 = self.padlayer(x_2)
        offsetx, offsety, offsetz = torch.meshgrid(
            [
                torch.arange(0, 2 * self.max_disp + 1),
                torch.arange(0, 2 * self.max_disp + 1),
                torch.arange(0, 2 * self.max_disp + 1),
            ],
            indexing="ij",
        )

        w, h, d = x_1.shape[2], x_1.shape[3], x_1.shape[4]
        x_out = torch.cat(
            [
                torch.mean(
                    x_1 * x_2[:, :, dx : dx + w, dy : dy + h, dz : dz + d],
                    1,
                    keepdim=True,
                )
                for dx, dy, dz in zip(
                    offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1)
                )
            ],
            1,
        )
        return x_out


class Encoder(nn.Module):
    """编码器中混合了Swin Transformer"""

    def __init__(self, in_channel=1, first_out_channel=8):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = ConvResBlock(in_channel, c)  # 1 c
        self.conv1 = ConvResBlock(c, 2 * c, stride=2)  # 1/2 2c
        self.conv2 = ConvResBlock(2 * c, 4 * c, stride=2)  # 1/4 4c

        # 1/4 2c
        self.trans = SwinTransformer(in_chans=in_channel, embed_dim=4 * c)
        self.up1 = UpConvBlock(4 * c, 4 * c)
        self.conv3_1 = ConvBlock(
            ndims=3,
            in_channels=6 * c,
            out_channels=2 * c,
            kernal_size=3,
            stride=1,
            padding=1,
        )
        self.conv3_2 = nn.Sequential(
            ConvInsBlock(2 * c, 4 * c, stride=2), ConvInsBlock(4 * c, 8 * c, stride=2)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1 c
        out1 = self.conv1(out0)  # 1/2 2c
        out2 = self.conv2(out1)  # 1/4 4c

        out_trans = self.up1(self.trans(x))  # 1/2 4c
        out3 = self.conv3_1(torch.cat([out_trans, out1], dim=1))  # 1/2 2c
        out3 = self.conv3_2(out3)  # 1/8 8c

        return [out0, out1, out2, out3]


class SDPM(nn.Module):
    """Step-wise Deformation Prediction Module
    Args:
        channels (int): Number of input channels.
        warp (SpatialTransformer): N-D Spatial Transformer
        diff (VecInt): Integrates a vector field via scaling and squaring.
        out_group (list): The group of output deformation fields, for example, [0,1,0,1] represents the second and fourth groups of output deformation fields
        corr (Correlation, optional): Used for correlation calculation
        n_groups (int, optional): Number of groups
        last_layer (bool, optional): is it the last layer, used to confirm whether to generate Ct at the same time as generating h
        use_corr (bool, optional): whether to use correlation calculation
        downsample_at (int, optional): in which group should the channel be halved, negative number indicate that the number of channel not be halved
    """

    def __init__(
        self,
        channels,
        warp,
        diff,
        out_group,
        corr=None,
        n_groups=4,
        last_layer=False,
        use_corr=True,
        downsample_at=3,
    ):
        super().__init__()
        assert n_groups is None or n_groups == len(out_group)
        n_groups = len(out_group)

        for i in out_group:
            assert i == 0 or i == 1
        assert out_group != []
        assert out_group[-1] == 1

        if use_corr:
            assert corr is not None

        assert (downsample_at < 0) or (
            downsample_at <= len(out_group) and downsample_at > 1
        )

        c = channels
        self.warp = warp
        self.diff = diff
        self.corr = corr
        self.last_layer = last_layer
        self.use_corr = use_corr

        self.out_group_index = []
        for i, j in enumerate(out_group):
            if j == 1:
                self.out_group_index.append(i)
        self.out_group_index = self.out_group_index[::-1]

        if use_corr:
            conv1 = nn.Sequential(
                ConvInsBlock(c + 27, 3 * c), ConvInsBlock(3 * c, c), ConvInsBlock(c, c)
            )
        else:
            conv1 = nn.Sequential(
                ConvInsBlock(c, c), ConvInsBlock(c, c), ConvInsBlock(c, c)
            )

        self.convMain = nn.ModuleList([conv1])
        curr_chanel = c
        for i in range(2, n_groups + 1):
            if downsample_at > 0 and i == downsample_at:
                self.convMain.append(
                    nn.Sequential(
                        ConvInsBlock(curr_chanel, curr_chanel // 2),
                        ConvInsBlock(curr_chanel // 2, curr_chanel // 2),
                    )
                )
                curr_chanel = curr_chanel // 2
            else:
                self.convMain.append(
                    nn.Sequential(
                        ConvInsBlock(curr_chanel, curr_chanel),
                        ConvInsBlock(curr_chanel, curr_chanel),
                    )
                )

        self.convLSTM = nn.ModuleList()
        self.convOut = nn.ModuleList()

        for i in self.out_group_index:
            lstm_in = c + c // 2 if (downsample_at < 0 or i < downsample_at - 1) else c

            self.convLSTM.append(ConvLSTM(lstm_in, c // 2))
            self.convOut.append(
                nn.Sequential(
                    ConvInsBlock(c // 2, c // 2),
                    nn.Conv3d(c // 2, 3, kernel_size=3, stride=1, padding=1),
                )
            )

        if last_layer:
            self.convCH = nn.Sequential(
                ConvInsBlock(curr_chanel, c),
                nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.convH = nn.Sequential(
                ConvInsBlock(curr_chanel, c // 2),
                nn.Conv3d(c // 2, c // 2, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, moving, fixed, Ct=None):
        warped = moving
        out_prev = None
        h = None
        for n, i in enumerate(self.out_group_index):
            if self.use_corr:
                x = torch.cat([warped, fixed, self.corr(warped, fixed)], dim=1)
            else:
                x = torch.cat([warped, fixed], dim=1)

            for j in range(i + 1):
                x = self.convMain[j](x)

            if n == 0:
                if self.last_layer:
                    Ch = self.convCH(x)
                    Ct, h = torch.split(Ch, [Ch.shape[1] // 2, Ch.shape[1] // 2], dim=1)
                else:
                    h = self.convH(x)

            Ct, h = self.convLSTM[n](Ct, h, x)
            out_curr = self.diff(self.convOut[n](h))

            if out_prev is not None:
                out_curr = self.warp(out_prev, out_curr) + out_curr

            warped = self.warp(moving, out_curr)  # TODO 修正点1，扭曲moving而不是warped
            out_prev = out_curr

        return out_curr, Ct


class SDTSNet(nn.Module):
    def __init__(self, inshape=(160, 192, 160), in_channel=1, channels=8):
        super().__init__()

        c = channels

        self.encoder = Encoder(in_channel, c)

        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
        for i in range(4):
            self.diff.append(VecInt([s // 2**i for s in inshape]))
        corr = Correlation()

        self.upsample_trilin = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )

        out_groups = [[0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]]

        self.defconv = nn.ModuleList()
        for i in range(4):
            self.defconv.append(
                SDPM(
                    channels=(2 ** (i + 1)) * c,
                    warp=self.warp[i],
                    diff=self.diff[i],
                    corr=corr,
                    last_layer=(i == 3),
                    out_group=out_groups[i],
                )
            )

        self.upconv = nn.ModuleList(
            [
                nn.Sequential(UpConvBlock(4 * c, 4 * c), nn.Conv3d(4 * c, c, 3, 1, 1)),
                nn.Sequential(
                    UpConvBlock(8 * c, 8 * c), nn.Conv3d(8 * c, 2 * c, 3, 1, 1)
                ),
                nn.Sequential(
                    UpConvBlock(8 * c, 8 * c), nn.Conv3d(8 * c, 4 * c, 3, 1, 1)
                ),
            ]
        )

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):

        # encoder stage
        M1, M2, M3, M4 = self.encoder(moving)
        F1, F2, F3, F4 = self.encoder(fixed)

        # decoder stage
        # layer 4
        flow, Ct1 = self.defconv[3](M4, F4)  # 16c 1/16  16c 1/16
        Ct1 = self.upconv[2](Ct1)  # 4c 1/4

        # layer 3
        flow = self.upsample_trilin(2 * flow)  # 3 1/4
        warped = self.warp[2](M3, flow)
        residual_flow, Ct2 = self.defconv[2](warped, F3, Ct1)  # 3 1/4  8c 1/4
        flow = self.warp[2](flow, residual_flow) + residual_flow
        Ct2 = self.upconv[1](torch.cat([Ct1, Ct2], dim=1))

        # layer 2
        flow = self.upsample_trilin(2 * flow)  # 3 1/2
        warped = self.warp[1](M2, flow)
        residual_flow, Ct3 = self.defconv[1](warped, F2, Ct2)  # 3 1/2  4c 1/2
        flow = self.warp[1](flow, residual_flow) + residual_flow
        Ct3 = self.upconv[0](torch.cat([Ct2, Ct3], dim=1))

        # layer 1
        flow = self.upsample_trilin(2 * flow)  # 3 1
        warped = self.warp[0](M1, flow)
        residual_flow, _ = self.defconv[0](warped, F1, Ct3)
        flow = self.warp[0](flow, residual_flow) + residual_flow

        y_moved = self.warp[0](moving, flow)

        return y_moved, flow


if __name__ == "__main__":
    torch.manual_seed(0)
    size = (1, 1, 80, 96, 80)
    model = SDTSNet(size[2:])

    A = torch.randn(size)
    B = torch.randn(size)

    # 前向传播
    out, flow = model(A, B)

    print(out.shape, flow.shape)
