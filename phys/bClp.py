from ..structwrapper import BinReader
import bpy

def ImportCLP(filepath):
    fr = open(filepath, "rb")

    f = BinReader(fr)
    if (f.read_u32() > 65535):
        f.update_endianess_flag(True)

    f.seek(0)
    clothheader = bpy.context.scene.bayo_clp_clothheader
    clothheader.m_num = f.read_s32()
    clothheader.m_limit_spring_rate = f.read_float32()
    clothheader.m_spd_rate = f.read_float32()
    clothheader.m_stretchy = f.read_float32()
    clothheader.m_bundle_num = f.read_s16()
    clothheader.m_bundle_num2 = f.read_s16()
    clothheader.m_thick = f.read_float32()
    clothheader.m_gravity_vec = [float(x) for x in f.read_f32_vector3()]
    clothheader.m_gravity_parts_no = f.read_u32()
    clothheader.m_first_bundle_rate = f.read_float32()
    clothheader.m_wind_vec = [float(x) for x in f.read_f32_vector3()]
    clothheader.m_wind_parts_no = f.read_s32()
    clothheader.m_wind_offset = [float(x) for x in f.read_f32_vector3()]
    clothheader.m_wind_sin = f.read_float32()
    clothheader.m_hit_adjust_rate = f.read_float32()


    cloth_wk = bpy.context.scene.bayo_clp_clothwk
    cloth_wk.clear()
    for i in range(clothheader.m_num):
        cloth_wk_item = cloth_wk.add()
        cloth_wk_item.no = str(f.read_s16())
        cloth_wk_item.no_up = str(f.read_s16())
        cloth_wk_item.no_down = str(f.read_s16())
        cloth_wk_item.no_side = str(f.read_s16())
        cloth_wk_item.no_poly = str(f.read_s16())
        cloth_wk_item.no_fix = str(f.read_s16())
        cloth_wk_item.rot_limit = f.read_float32()
        cloth_wk_item.offset = [float(x) for x in f.read_f32_vector3()]