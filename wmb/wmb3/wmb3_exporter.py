from ...structwrapper import BinWriter
import bpy
import struct

def export(filepath):
    rf = open(filepath, "wb")
    f = BinWriter(rf)

    # Worst exporter ever, literally just to get custom models going

    f.write(b"WMB3")
    f.write_u32(65536) # 0.1
    f.write_s32(0)
    f.write_s32(65847)
    f.write_s16(0)
    f.write_s16(-1)

    f.write_float32(0)
    f.write_float32(4.4813)
    f.write_float32(-6.4738)
    f.write_float32(2.1057)
    f.write_float32(4.4862)
    f.write_float32(7.6531)

    f.write_u32(128)

    wmb_collection =  bpy.context.view_layer.layer_collection.children["WMB"]
    sub_collection = [x for x in wmb_collection.children if x.is_visible][0]
    arm_obj = sub_collection.collection.objects[0]

