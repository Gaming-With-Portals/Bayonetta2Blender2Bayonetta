import bpy
from ..utils.util import BayonettaVector4Property

class BayoMaterialPanelAdvanced(bpy.types.Panel):
    bl_label = "Advanced Properties"
    bl_idname = "MATERIAL_PT_bayo_material_adv"
    bl_parent_id = "MATERIAL_PT_bayo_material" 
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"
    bl_options = {'DEFAULT_CLOSED'} 

    def draw(self, context):
        layout = self.layout
        mat = context.material
        if not mat or not hasattr(mat, "bayo_data"):
            layout.label(text="No Bayonetta material data found.")
            return

        box = layout.box()
        for param in mat.bayo_data.parameters:
            
            if param.type == "f4_float3_t":
                box.label(text=param.name)
                box.prop(param, "value_vec3", text="")
            elif param.type == "f4_float2_t":
                box.label(text=param.name)
                box.prop(param, "value_vec2", text="")
            elif param.type == "f4_float_t":
                box.label(text=param.name)
                box.prop(param, "value_float", text="")
            elif param.type == "f4_ignored_t":
                continue
            elif param.type in {"sampler2D_t", "samplerCUBE_t"}:
                continue
            else:
                box.label(text=param.name)
                box.prop(param, "value_vec4", text="")

class BayoMaterialPanel(bpy.types.Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = 'material'
    bl_label = "Bayonetta Material Properties"
    bl_idname = "MATERIAL_PT_bayo_material"

    def draw(self, context):
        layout = self.layout
        mat = context.material

        if mat is None:
            layout.label("Select a material first")
            return

        layout.prop(mat.bayo_data, "type")
        layout.prop(mat.bayo_data, "flags")
        layout.prop(mat.bayo_data, "bayonetta_2")

        box = layout.box()
        for param in mat.bayo_data.parameters:
            if param.type in {"sampler2D_t", "samplerCUBE_t"}:
                box.prop(param, "value_int", text=param.name)


        
