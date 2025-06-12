import os
import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from ...utils import ioUtils as io

class WTAData:
    def __init__(self, f, wtpFile) -> None:
        # Header
        self.magic = f.read(4)
        self.version = io.read_uint32(f)
        self.num_files = io.read_uint32(f)
        self.offsetTextureOffsets = io.read_uint32(f)
        self.offsetTextureSizes = io.read_uint32(f)
        self.offsetTextureFlags = io.read_uint32(f)
        self.offsetTextureIdx = io.read_uint32(f)
        self.offsetTextureInfo = io.read_uint32(f)

        # Texture Offsets
        f.seek(self.offsetTextureOffsets)
        self.offsets = []
        for i in range(self.num_files):
            self.offsets.append(io.read_uint32(f))
        
        # Texture Sizes
        f.seek(self.offsetTextureSizes)
        self.sizes = []
        for i in range(self.num_files):
            self.sizes.append(io.read_uint32(f))

        # Texture Flags
        f.seek(self.offsetTextureFlags)
        self.flags = []
        for i in range(self.num_files):
            self.flags.append(io.read_uint32(f))

        self.idx = []
        for i in range(self.num_files):
            self.idx.append(i)

        """# Texture Idx
        f.seek(self.offsetTextureIdx)
        self.idx = []
        for i in range(self.num_files):
            self.idx.append(io.read_uint32(f))
        """ # Not in bayonetta 1
        # Texture Info
        """
        self.infos = []
        for i in range(self.num_files):
            info = []
            info.append(io.read_uint32(f))
            info.append([io.read_uint32(f) for x in range(4)])
            self.infos.append(info)
        """

        self.data = wtpFile.read()

        self.wtaPath = f.name

    def  extract_textures(self, extractionDir):
        count = 0
        fileName = os.path.basename(self.wtaPath)
        dir = os.path.dirname(self.wtaPath)
        for i in range(self.num_files):
            os.makedirs(extractionDir, exist_ok=True)
            with open(os.path.join(extractionDir, f"{self.idx[i]:0>8X}.dds"), "wb") as f:
                f.write(self.data[self.offsets[i]:self.offsets[i]+self.sizes[i]])
            count += 1
        return count

def extractFromWta(wtaPath, wtpPath, extractDir) -> int:
    with (
        open(wtaPath, "rb") as wtaFile,
        open(wtpPath, "rb") as wtpFile
    ):
        wta = WTAData(wtaFile, wtpFile)
    extractedCount = wta.extract_textures(extractDir)
    return extractedCount
