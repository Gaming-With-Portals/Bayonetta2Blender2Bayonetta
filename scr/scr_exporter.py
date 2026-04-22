import bpy
from ..wmb import wmb_exporter
import os
import struct

def export(filepath):
    ALIGN_TARGET = 64
    tmp_dir = os.path.join(os.path.dirname(filepath), "b2b_scr", os.path.basename(filepath))
    os.makedirs(tmp_dir, exist_ok=True)

    scr_collection =  bpy.context.view_layer.layer_collection.children["SCR"]
    for child in scr_collection.children:
        wmb_exporter.export(os.path.join(tmp_dir, child.name+".wmb"), static_mesh=True, targetCol=child, copy_uv=False)


    packinglist = []
    for wmb in os.listdir(tmp_dir):
        friendName = wmb_exporter.decimalFixup(os.path.splitext(wmb)[0])
        packinglist.append((os.path.join(tmp_dir, wmb), friendName, os.path.getsize(os.path.join(tmp_dir, wmb))))
        if (len(friendName) > 16):
            print(f"SCR name of {friendName} is more than 16 bytes!\nThis won't work in Bayonetta!")
            return {'CANCELLED'}

    f = open(filepath, "wb")
    f.write(b"SCR\x00")
    f.write(struct.pack("<I", len(packinglist)))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 1))

    offsets = []
    offset_ticker = wmb_exporter.align(140*len(packinglist), ALIGN_TARGET)
    for mdl in packinglist:
        rec_start = f.tell()
        f.write(mdl[1].ljust(16, '\x00').encode())
        f.write(struct.pack("<I", offset_ticker-rec_start))

        f.write(struct.pack("<f", 0))
        f.write(struct.pack("<f", 0))
        f.write(struct.pack("<f", 0))

        f.write(struct.pack("<f", 0))
        f.write(struct.pack("<f", 0))
        f.write(struct.pack("<f", 0))

        f.write(struct.pack("<f", 1))
        f.write(struct.pack("<f", 1))
        f.write(struct.pack("<f", 1))

        f.write(struct.pack("<h", -1))
        for i in range(7):
            f.write(struct.pack("<h", 0))

        f.write(struct.pack("<I", 0))


        for i in range(10):
            f.write(struct.pack("<h", -1))
        for i in range(22):
            f.write(struct.pack("<h", 0))

        offsets.append(offset_ticker)
        offset_ticker+=wmb_exporter.align(mdl[2], 2048)

    for i, mdl in enumerate(packinglist):
        x = open(mdl[0], "rb")
        f.seek(offsets[i])
        f.write(x.read())
        x.close()

    wtb_start = wmb_exporter.align(f.tell(), 2048)
    f.seek(wtb_start)
    if ("textures" in scr_collection.collection):
        x = open(scr_collection.collection["textures"], "rb")
        f.write(x.read())
        x.close()
    else:
        f.write(b"WTB\x00")
        for i in range(10):
            f.write(struct.pack("<I", 0))

    f.seek(0x8)
    f.write(struct.pack("<I", wtb_start))


    return {'FINISHED'}

    