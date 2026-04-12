from ..structwrapper import BinReader
import os
import struct
from . import swizzle as deswizzler
import json
import subprocess
import bpy
from ..consts import ADDON_NAME
# Credit: https://github.com/aboood40091/BNTX-Extractor/blob/master/bntx_extract.py

astc_enc_path = ""





print(f"ASTC Encoder Registered as {astc_enc_path}")

blk_dims = {  # format -> (blkWidth, blkHeight)
    0x1a: (4, 4), 0x1b: (4, 4), 0x1c: (4, 4),
    0x1d: (4, 4), 0x1e: (4, 4), 0x1f: (4, 4),
    0x20: (4, 4), 0x2d: (4, 4), 0x2e: (5, 4),
    0x2f: (5, 5), 0x30: (6, 5),
    0x31: (6, 6), 0x32: (8, 5),
    0x33: (8, 6), 0x34: (8, 8),
    0x35: (10, 5), 0x36: (10, 6),
    0x37: (10, 8), 0x38: (10, 10),
    0x39: (12, 10), 0x3a: (12, 12),
}

bpps = {  # format -> bytes_per_pixel
    0x0b: 0x04, 0x07: 0x02, 0x02: 0x01, 0x09: 0x02, 0x1a: 0x08,
    0x1b: 0x10, 0x1c: 0x10, 0x1d: 0x08, 0x1e: 0x10, 0x1f: 0x10,
    0x20: 0x10, 0x2d: 0x10, 0x2e: 0x10, 0x2f: 0x10, 0x30: 0x10,
    0x31: 0x10, 0x32: 0x10, 0x33: 0x10, 0x34: 0x10, 0x35: 0x10,
    0x36: 0x10, 0x37: 0x10, 0x38: 0x10, 0x39: 0x10, 0x3a: 0x10,
}

dds_formats = {
    0x0b01: 'R8_G8_B8_A8_UNORM',
    0x0b06: 'R8_G8_B8_A8_SRGB',
    0x0701: 'R5_G6_B5_UNORM',
    0x0201: 'R8_UNORM',
    0x0901: 'R8_G8_UNORM',
    0x1a01: 'BC1_UNORM',
    0x1a06: 'BC1_SRGB',
    0x1b01: 'BC2_UNORM',
    0x1b06: 'BC2_SRGB',
    0x1c01: 'BC3_UNORM',
    0x1c06: 'BC3_SRGB',
    0x1d01: 'BC4_UNORM',
    0x1d02: 'BC4_SNORM',
    0x1e01: 'BC5_UNORM',
    0x1e02: 'BC5_SNORM',
    0x1f01: 'BC6H_UF16',
    0x1f02: 'BC6H_SF16',
    0x2001: 'BC7_UNORM',
    0x2006: 'BC7_SRGB',
}

astc_blocksize = {
    0x2d01: (4, 4),
    0x2d06: (4, 4),
    0x2e01: (5, 4),
    0x2e06: (5, 4),
    0x2f01: (5, 5),
    0x2f06: (5, 5),
    0x3001: (6, 5),
    0x3006: (6, 5),
    0x3101: (6, 6),
    0x3106: (6, 6),
    0x3201: (8, 5),
    0x3206: (8, 5),
    0x3301: (8, 6),
    0x3306: (8, 6),
    0x3401: (8, 8),
    0x3406: (8, 8),
    0x3501: (10, 5),
    0x3506: (10, 5),
    0x3601: (10, 6),
    0x3606: (10, 6),
    0x3701: (10, 8),
    0x3706: (10, 8),
    0x3801: (10, 10),
    0x3806: (10, 10),
    0x3901: (12, 10),
    0x3906: (12, 10),
    0x3a01: (12, 12),
    0x3a06: (12, 12)
}

astc_formats = {
    0x2d01: 'ASTC4x4',
    0x2d06: 'ASTC4x4 SRGB',
    0x2e01: 'ASTC5x4',
    0x2e06: 'ASTC5x4 SRGB',
    0x2f01: 'ASTC5x5',
    0x2f06: 'ASTC5x5 SRGB',
    0x3001: 'ASTC6x5',
    0x3006: 'ASTC6x5 SRGB',
    0x3101: 'ASTC6x6',
    0x3106: 'ASTC6x6 SRGB',
    0x3201: 'ASTC8x5',
    0x3206: 'ASTC8x5 SRGB',
    0x3301: 'ASTC8x6',
    0x3306: 'ASTC8x6 SRGB',
    0x3401: 'ASTC8x8',
    0x3406: 'ASTC8x8 SRGB',
    0x3501: 'ASTC10x5',
    0x3506: 'ASTC10x5 SRGB',
    0x3601: 'ASTC10x6',
    0x3606: 'ASTC10x6 SRGB',
    0x3701: 'ASTC10x8',
    0x3706: 'ASTC10x8 SRGB',
    0x3801: 'ASTC10x10',
    0x3806: 'ASTC10x10 SRGB',
    0x3901: 'ASTC12x10',
    0x3906: 'ASTC12x10 SRGB',
    0x3a01: 'ASTC12x12',
    0x3a06: 'ASTC12x12 SRGB'
}

def extractBntx(filePath, outputDir):
   
    bntx = open(filePath, "rb")
    f = BinReader(bntx)

    magic = f.read(8)
    version = f.read_u32()
    bom = f.read_u16()
    alignment = f.read_u8()
    unknown = f.read_u8()
    fileNameOffset = f.read_u32()
    realloc = f.read_u32()
    offsetBinaryBlock = f.read_u32()
    fileSize = f.read_u32()
    
    nx_magic = f.read(4)
    if (nx_magic != b"NX  "):
        print("Failed to load BNTX! (Invalid NX Chunk)")
        return
    
    f.read_u32()
    brtiOffsetOffset = f.read_u32()
    f.read_u32()
    fileOffset = f.read_u32()

    f.seek(brtiOffsetOffset)
    brtiOffset = f.read_u32()

    
    f.seek(brtiOffset)
    if (f.read(4) != b"BRTI"):
        print("Failed to load BNTX! (Invalid BRTI Chunk)")
        return
    
    f.advance(12)
    tileMode = f.read_u8()
    dim = f.read_u8()
    flags = f.read_u16()
    swizzle = f.read_u16()
    numMips = f.read_u16()
    unk18 = f.read_u32()
    fmt = f.read_u32()
    unk20 = f.read_u32()
    width = f.read_u32()
    height = f.read_u32()
    unk2C = f.read_u32()
    numFaces = f.read_u32()
    sizeRange = f.read_u32()
    unk38 = f.read_u32()
    unk3C = f.read_u32()
    unk40 = f.read_u32()
    unk44 = f.read_u32()
    unk48 = f.read_u32()
    unk4C = f.read_u32()
    imageSize = f.read_u32()
    brti_alignment = f.read_u32()

    if (fmt in dds_formats):
        print(f"BNTX Format: {dds_formats[fmt]} (DXT)")
        
        x = open(outputDir+".dds", "wb")
        x.write(b"DDS ")
        x.write(struct.pack("<I", 124))
        x.write(struct.pack("<I", 528391))
        x.write(struct.pack("<I", width))
        x.write(struct.pack("<I", height))
        x.write(struct.pack("<I", 262144))
        x.write(struct.pack("<I", 0))
        x.write(struct.pack("<I", numMips))
        pos = x.tell()
        x.write(b"BAYONETTA2BLENDER")
        x.seek(pos+44)
        x.write(struct.pack("<I", 32))
        x.write(struct.pack("<I", 4))

        requireDXGI = False
        if (dds_formats[fmt].startswith("BC1")):
            x.write(b"DXT1")
        elif (dds_formats[fmt].startswith("BC2")):
            x.write(b"DXT3")
        elif (dds_formats[fmt].startswith("BC3")):
            x.write(b"DXT5")
        elif (dds_formats[fmt].startswith("BC4")):
            x.write(b"ATI1")
        elif (dds_formats[fmt].startswith("BC5")):
            x.write(b"ATI2")
        elif (dds_formats[fmt].startswith("BC7")):
            requireDXGI=True
            x.write(b"DX10")


        x.write(struct.pack("<I", 0))
        x.write(struct.pack("<I", 0))
        x.write(struct.pack("<I", 0))
        x.write(struct.pack("<I", 0))
        x.write(struct.pack("<I", 0))
        x.write(struct.pack("<I", 4096))

        if (requireDXGI):
            x.seek(128)
            if (dds_formats[fmt] == 'BC7_UNORM'):
                x.write(struct.pack("<I", 98))
            elif (dds_formats[fmt] == 'BC7_SRGB'):
                x.write(struct.pack("<I", 99))
            elif (dds_formats[fmt] == 'BC6H_UF16'):
                x.write(struct.pack("<I", 95))
            elif (dds_formats[fmt] == 'BC6H_SF16'):
                x.write(struct.pack("<I", 96))
            x.write(struct.pack("<I", 3))
            x.write(struct.pack("<I", 0))
            x.write(struct.pack("<I", 1))
            x.write(struct.pack("<I", 0))



        if (fmt >> 8) in blk_dims:
            blkWidth, blkHeight = blk_dims[fmt >> 8]

        else:
            blkWidth, blkHeight = 1, 1

        bpp = bpps[fmt >> 8]

        if (requireDXGI):
            x.seek(148)
        else:
            x.seek(128)
        

        f.seek(fileOffset+0x10)

        unswizzled = deswizzler.deswizzle(width, height, blkWidth, blkHeight, bpp, tileMode, brti_alignment, sizeRange, f.read(imageSize))

        x.write(unswizzled)

    elif (fmt in astc_formats):
        print(f"BNTX Format: {astc_formats[fmt]} (ASTC)")
        x = open(outputDir+".astc", "wb")
        x.write(struct.pack("<I", 1554098963)) # Magic
        x.write(struct.pack("<B", astc_blocksize[fmt][0]))  
        x.write(struct.pack("<B", astc_blocksize[fmt][1]))
        x.write(struct.pack("<B", 1))
        x.write(struct.pack("<I", width)[0:3])
        x.write(struct.pack("<I", height)[0:3])
        x.write(struct.pack("<I", 1)[0:3])

        blkWidth, blkHeight = astc_blocksize[fmt]
        bpp = bpps[fmt >> 8]
        size = ((width + blkWidth - 1) // blkWidth) * ((height + blkHeight - 1) // blkHeight) * bpp

        f.seek(fileOffset+0x10)
        unswizzled = deswizzler.deswizzle(width, height, blkWidth, blkHeight, bpp, tileMode, brti_alignment, sizeRange, f.read(imageSize))
        unswizzled = unswizzled[:size]

        x.seek(0x10)
        
        x.write(unswizzled)
        x.close()

        if (astc_enc_path == ""):
            print("[!] Missing AstcEnc.exe, the ASTC file is ready but will not be autoconverted to png.")
            return
        
        result = subprocess.run(
            [astc_enc_path, "-dl", outputDir+".astc", outputDir+".png"],
        )


    else:
        print("Unsupported BNTX format!")
        return





def extractTextures(wtaFilePath, wtpFilePath, targetDirectory):
    global astc_enc_path
    prefs = bpy.context.preferences.addons[ADDON_NAME].preferences
    astc_enc_path = prefs.astcEncDir


    wta = open(wtaFilePath, "rb")
    f = BinReader(wta)

    if (f.read(4) != b"WTB\x00"):
        print("Not a valid texture package.")
        return 
    
    version = f.read_u32()
    textureCount = f.read_u32()
    offsetTableOffset = f.read_u32()
    sizeTableOffset = f.read_u32()
    flagTableOffset = f.read_u32()
    idxTableOffset = f.read_u32()
    infoTableOffset = f.read_u32()
    mipmapTableOffset = f.read_u32()

    f.seek(offsetTableOffset)
    offsets = f.read_u32_array(textureCount)

    f.seek(sizeTableOffset)
    sizes = f.read_u32_array(textureCount)

    f.seek(flagTableOffset)
    flags = f.read_u32_array(textureCount)

    if (version == 1):
        f.seek(idxTableOffset)
        ids = f.read_u32_array(textureCount)
    else:
        ids = range(textureCount)

    wta.close()
    os.makedirs(targetDirectory, exist_ok=True)
    wtp = open(wtpFilePath, "rb")
    for i in range(textureCount):
        wtp.seek(offsets[i])
        ext = ".bin"
        fileMagic = wtp.read(4)
        if (fileMagic == b"DDS\x00"):
            ext=".dds"
        dir = os.path.join(targetDirectory, f"{ids[i]:0>8X}{ext}")

        with open(dir, "wb") as f:
            wtp.seek(offsets[i])
            f.write(wtp.read(sizes[i]))

        if (fileMagic == b"BNTX"):
            print("Attempting to extract BNTX...")
            extractBntx(os.path.join(targetDirectory, f"{ids[i]:0>8X}{ext}"), os.path.join(targetDirectory, f"{ids[i]:0>8X}"))


    wtp.close()


    for f in os.listdir(targetDirectory):
        if (os.path.splitext(f)[1]==".bin"):
            os.remove(os.path.join(targetDirectory, f))