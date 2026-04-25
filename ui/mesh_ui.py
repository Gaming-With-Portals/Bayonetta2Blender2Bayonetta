import bpy

class BayoObjectPanel(bpy.types.Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = 'object'
    bl_label = "Bayonetta Batch Properties"
    bl_idname = "MATERIAL_PT_bayo_object"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.active_object.bayo_data, "flags")