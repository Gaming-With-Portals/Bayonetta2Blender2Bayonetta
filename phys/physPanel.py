import os, bpy
from bpy.props import StringProperty
from .propertyGroups import ClothHeader, ClothWK
from bpy_extras.io_utils import ExportHelper, ImportHelper
from .bClp import ImportCLP

class B2BOpenClpFile(bpy.types.Operator, ImportHelper):
    bl_idname = "b2b.open_clp_file"
    bl_label = "Open CLP File"
    bl_options = {"UNDO"}
    filter_glob: StringProperty(default="*.clp", options={'HIDDEN'})

    def execute(self, context):
        ImportCLP(self.filepath)
        return {"FINISHED"}




class B2BPhysicsEditor(bpy.types.Panel):
    bl_label = "Bayonetta Physics Editor"
    bl_idname = "B2B_PT_PhysicsEditorTop"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BAYO: Physics Editor"

    def draw(self, context):
        return
    

class B2BCLPEditor(bpy.types.Panel):
    bl_label = "CLP Editor"
    bl_idname = "b2B_PT_CLPEdit"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = B2BPhysicsEditor.bl_idname
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("b2b.open_clp_file", text="Open CLP File")

class B2BCLPHeader(bpy.types.Panel):
    bl_label = "CLP Header"
    bl_idname = "B2B_PT_CLPHead"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = B2BCLPEditor.bl_idname

    def draw(self, context):
        layout = self.layout
        layout.label(text="CLOTH_HEADER")
        clothheader = bpy.context.scene.bayo_clp_clothheader
        layout.prop(clothheader, "m_limit_spring_rate")
        layout.prop(clothheader, "m_spd_rate")
        layout.prop(clothheader, "m_stretchy")
        layout.prop(clothheader, "m_bundle_num")
        layout.prop(clothheader, "m_bundle_num2")
        layout.prop(clothheader, "m_thick")
        layout.prop(clothheader, "m_gravity_vec")
        layout.prop(clothheader, "m_gravity_parts_no")
        layout.prop(clothheader, "m_first_bundle_rate")
        layout.prop(clothheader, "m_wind_vec")
        layout.prop(clothheader, "m_wind_parts_no")
        layout.prop(clothheader, "m_wind_offset")
        layout.prop(clothheader, "m_wind_sin")

class B2BCLPList(bpy.types.Panel):
    bl_label = "CLP List"
    bl_idname = "B2B_PT_CLPList"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = B2BCLPEditor.bl_idname

    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        #row.operator("clp.add_clothwk")
        row = layout.row()
        #row.operator("clp.auto_generate_clothwk_selected")

        row = layout.row()
        #row.prop(search_options, "filter_by_selected")

        layout.label(text="CLOTH_WK_LIST")
        cloth_wk = bpy.context.scene.bayo_clp_clothwk
        for index, item in enumerate(cloth_wk):
            '''if search_options.filter_by_selected:
                if item.no not in selected_bone_ids:
                    continue'''

            box = layout.box()

            # Align label to left and remove button to right
            row = box.row(align=True)
            row.label(text=item.no)
            row.operator("clp.remove_clothwk", text="", icon='X', emboss=False).index = index

            row = box.row()
            row.prop(item, "no")
            row = box.row()
            row.prop(item, "no_up")
            row = box.row()
            row.prop(item, "no_down")
            row = box.row()
            row.prop(item, "no_side")
            row.prop(item, "no_poly")
            row = box.row()
            row.prop(item, "no_fix")
            row = box.row()
            row.prop(item, "offset")
            row = box.row()
            row.prop(item, "rot_limit")
            row.prop(item, "m_original_rate")





physicsclasses = [
    B2BPhysicsEditor,
    B2BCLPEditor,
    B2BCLPHeader,
    B2BOpenClpFile,
    B2BCLPList,
    ClothWK,
    ClothHeader
]

def register():
    for cls in physicsclasses:
        bpy.utils.register_class(cls)

    bpy.types.Scene.bayo_clp_clothwk = bpy.props.CollectionProperty(type=ClothWK)
    bpy.types.Scene.bayo_clp_clothheader = bpy.props.PointerProperty(type=ClothHeader)

def deregister():
    for cls in physicsclasses:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.bayo_clp_clothwk
    del bpy.types.Scene.bayo_clp_clothheader