from ..structwrapper import BinReader
from ..wmb import wmb_importer
import bpy
import os
import mathutils, math
from ..wta_wtp import pg_texture


def ImportSCR(filepath):
    rawf = open(filepath, "rb")
    extracted_texture_cmn = os.path.join(os.path.dirname(filepath), "textures")
    extracted_wmb_dir = os.path.join(os.path.dirname(filepath), "b2b_scr_extracted", os.path.basename(filepath))
    extracted_textures_dir = os.path.join(extracted_wmb_dir, "textures")
    os.makedirs(extracted_wmb_dir, exist_ok=True)


    f = BinReader(rawf)

    if not (f.read(4).decode() == "SCR\x00"):
        print("Not an SCR!")
        return {'CANCELLED'}
    
    print("Importing SCR...")

    scr_type = "B1"

    if ("SCR" not in bpy.data.collections):
        scr_collection = bpy.data.collections.new("SCR")
        bpy.context.scene.collection.children.link(scr_collection)
    else:
        scr_collection = bpy.data.collections["SCR"]

    f.seek(0x4)
    ver = f.read_u16()
    if (ver == 18):
        f.seek(0xC)
        if (f.read_u32() != 1):
            scr_type = "B2"


    f.seek(0x4)
    if (scr_type == "B1"):
        print("SCR Type: Bayonetta 1")
        model_count = f.read_u32()
        offset_textures = f.read_u32()
        tag = f.read_u32()
        if (tag!=1):
            print("WARNING! Odd tag for Bayonetta 1!")

        wmb_offsets = []
        f.seek(0x10)
        for i in range(model_count):
            f.seek(0x14+(i*140))
            wmb_offsets.append(f.read_u32())
        wmb_offsets.append(offset_textures)

        f.seek(0, 2)
        eof = f.tell()
        f.seek(offset_textures)
        txPk = open(os.path.join(extracted_wmb_dir,"scr.wtb"), "wb")
        txPk.write(f.read(offset_textures-eof))
        txPk.close()
        pg_texture.extractTextures(os.path.join(extracted_wmb_dir,"scr.wtb"), os.path.join(extracted_wmb_dir,"scr.wtb"), extracted_textures_dir)
        scr_collection["textures"]=os.path.join(extracted_wmb_dir,"scr.wtb")
        


        f.seek(0x10)
        for i in range(model_count):
            f.seek(0x10+(i*140))
            scr_name = f.read(16).decode().replace("\x00", "")
            offset = f.read_u32() + 0x10+(i*140)
            translation = f.read_f32_vector3()
            rotation = f.read_f32_vector3()
            scale = f.read_f32_vector3()
            paramsA = f.read_s16_array(8)
            flags = str(f.read_u32())
            paramsB = f.read_s16_array(32)





            f.seek(offset)
            wmb_data = f.read(wmb_offsets[i+1]-offset)
            wmb_path = os.path.join(extracted_wmb_dir, f"{scr_name}.wmb")
            x = open(os.path.join(extracted_wmb_dir, f"{scr_name}.wmb"), "wb")
            x.write(wmb_data)
            x.close()

            wmb_importer.ImportWMB(wmb_path, extracted_textures_dir, False, False, target_col="SCR")

            model_collection = bpy.data.collections.get(scr_name)
            model_collection["flags"] = flags
            model_collection["paramsA"] = paramsA
            model_collection["paramsB"] = paramsB

            for obj in model_collection.objects:
                if obj.type == 'MESH':
                    obj.location = mathutils.Vector(translation)
                    obj.scale = mathutils.Vector(scale)
    if (scr_type == "B2"):
        print("SCR Type: Bayonetta 2")
        f.read_u16()

        model_count = f.read_u16()
        offset_models = f.read_u32()

        f.seek(offset_models)
        model_offsets = []
        for i in range(model_count):
            model_offsets.append(f.read_u32())




        wmb_offsets = []
        for i in range(model_count):
            f.seek(model_offsets[i])
            wmb_offsets.append(f.read_u32())
        f.seek(0, 2)
        wmb_offsets.append(f.tell())


        for i in range(model_count):
            f.seek(model_offsets[i])
            offset = f.read_u32()
            scr_name = f.read(64).decode().replace("\x00", "")

            translation = f.read_f32_vector3()
            rotation = f.read_f32_vector3() # we dont know how to parse this
            scale = f.read_f32_vector3()
            params = f.read_s16_array(18)






            f.seek(offset)
            wmb_data = f.read(wmb_offsets[i+1]-offset)
            wmb_path = os.path.join(extracted_wmb_dir, f"{scr_name}.wmb")
            x = open(os.path.join(extracted_wmb_dir, f"{scr_name}.wmb"), "wb")
            x.write(wmb_data)
            x.close()

            wmb_importer.ImportWMB(wmb_path, extracted_texture_cmn, False, False, target_col="SCR")

            model_collection = bpy.data.collections.get(scr_name)
            model_collection["params"] = params

            for obj in model_collection.objects:
                if obj.type == 'MESH':
                    obj.location = mathutils.Vector(translation)
                    obj.scale = mathutils.Vector(scale)



    return {'FINISHED'}
