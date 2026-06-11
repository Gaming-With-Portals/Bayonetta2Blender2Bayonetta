import bpy

bone_items = [("4095", "None", "")]

def clp_bone_items(self, context):
    return bone_items

class ClothHeader(bpy.types.PropertyGroup):
    m_num : bpy.props.IntProperty()
    m_limit_spring_rate : bpy.props.FloatProperty()
    m_spd_rate : bpy.props.FloatProperty()
    m_stretchy : bpy.props.FloatProperty()
    m_bundle_num : bpy.props.IntProperty()
    m_bundle_num2 : bpy.props.IntProperty()
    m_thick : bpy.props.FloatProperty()
    m_gravity_vec : bpy.props.FloatVectorProperty()
    m_gravity_parts_no : bpy.props.IntProperty()
    m_first_bundle_rate : bpy.props.FloatProperty()
    m_wind_vec : bpy.props.FloatVectorProperty()
    m_wind_parts_no : bpy.props.IntProperty()
    m_wind_offset : bpy.props.FloatVectorProperty()
    m_wind_sin : bpy.props.FloatProperty()
    m_hit_adjust_rate : bpy.props.FloatProperty()

class ClothWK(bpy.types.PropertyGroup):
    no : bpy.props.EnumProperty(items=clp_bone_items, default=0)
    no_up : bpy.props.EnumProperty(items=clp_bone_items, default=4095)
    no_down : bpy.props.EnumProperty(items=clp_bone_items, default=4095)
    no_side : bpy.props.EnumProperty(items=clp_bone_items, default=4095)
    no_poly : bpy.props.EnumProperty(items=clp_bone_items, default=4095)
    no_fix : bpy.props.EnumProperty(items=clp_bone_items, default=4095)

    rot_limit : bpy.props.FloatProperty(default=0.785)
    offset : bpy.props.FloatVectorProperty(default=(0, -0.1, 0))