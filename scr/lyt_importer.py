from ..structwrapper import BinReader
from ..wmb import wmb_importer
import bpy
import os
import mathutils

def ImportLYT(filepath):
    rawf = open(filepath, "rb")
    f = BinReader(rawf)

    if (f.read(4).decode() != "LYT\x00"):
        print("Invalid LYT! (Bad magic)")
        return
    
    version = f.read_u16()
    unknown = f.read_u16()
    modelCount = f.read_u32()
    unknownOffset = f.read_u32()

    if ("LYT" not in bpy.data.collections):
        scr_collection = bpy.data.collections.new("LYT")
        bpy.context.scene.collection.children.link(scr_collection)
    else:
        scr_collection = bpy.data.collections["LYT"]

    f.seek(0x10)
    from ..dat_dtt.importer.datImportOperator import ImportData

    missings = []

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))
    for i in range(modelCount):
        id = f.read_u32()
        folder = f.read(8).decode().replace("\x00", "")
        nameHead = f.read(2).decode().replace("\x00", "")
        id = f.read_u16()
        transformOffset = f.read_u32()

        pos = f.tell()
        f.seek(transformOffset)
        tx, ty, tz = f.read_f32_vector3()
        translation = (tx, -tz, ty)
        scale = f.read_f32_vector3()

        datPath = f"{base}/{folder}/{nameHead}{id:04x}.dat"

        if (os.path.exists(datPath)):
            ImportData(False, datPath, isStageSubmesh=True)

            model_collection = bpy.data.collections.get(f"{nameHead}{id:04x}")
            if (model_collection):
                for obj in model_collection.objects:
                    if obj.type == 'MESH':
                        obj.location = mathutils.Vector(translation)
                        obj.scale = mathutils.Vector(scale)
        else:
            missings.append(datPath)


        f.seek(pos)
    
    print("LYT import done!")
    for missing in missings:
        print(f"Missing: {missing}")