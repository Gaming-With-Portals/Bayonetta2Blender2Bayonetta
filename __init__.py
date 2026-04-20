bl_info = {
    "name": "Bayonetta2Blender2Bayonetta",
    "author": "Gaming With Portals, Raq (With some things based off MGR2Blender2MGR)",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "description": "Import/Export Various Bayonetta Data files.",
    "category": "Import-Export"}


import bpy
import os
from bpy.app.handlers import persistent
from .wmb.wmbImportOperator import ImportBayoWMB
from .wmb.wmbImportOperator import ExportBayoWMB
from .dat_dtt.importer.datImportOperator import ImportNierDat
from .ui.material_ui import BayoMaterialPanel, BayoMaterialToJSON, BayoJSONToMaterial
from .ui.material_ui import BayoMaterialPanelAdvanced
from .ui.mesh_ui import BayoObjectPanel
from .utils.util import BayonettaVector4Property
from .wmb.wmb_materials import BayonettaParameter, BayonettaTexture, Bayonetta2Data, BayonettaExMaterialData
from .wmb.wmb_materials import BayoMaterialDataProperty
from .scr.scrOperators import ImportBayoSCR, ImportVanqLYT
from .utils.utilOperators import RipMeshByUVIslands, RemoveUnusedVertexGroups

class BayonettaObjectMenu(bpy.types.Menu):
    bl_idname = 'OBJECT_MT_b2b2b'
    bl_label = 'Bayonetta Tools'
    def draw(self, context):
        self.layout.operator(RemoveUnusedVertexGroups.bl_idname, icon="GROUP_VERTEX")
        self.layout.operator(RipMeshByUVIslands.bl_idname, icon="UV_ISLANDSEL")

preview_collections = {}

class IMPORT_BN_MainMenu(bpy.types.Menu):
    bl_label = "Bayonetta"
    bl_idname = "IMPORT_BN_main_menu"

    def draw(self, context):
        pcoll = preview_collections["main"]
        raiden_icon = pcoll["bayo"] 
        vanquish_icon = pcoll["vanq"] 
   
        self.layout.operator(ImportNierDat.bl_idname, text="Archive File (.dat, .dtt)", icon_value=raiden_icon.icon_id)
        self.layout.operator(ImportBayoWMB.bl_idname, text="Model File (.wmb)", icon_value=raiden_icon.icon_id)
        self.layout.operator(ImportBayoSCR.bl_idname, text="Stage File (.scr)", icon_value=raiden_icon.icon_id)
        #self.layout.operator(ImportVanqLYT.bl_idname, text="Vanquish Stage (.lyt)")


class EXPORT_BN_MainMenu(bpy.types.Menu):
    bl_label = "Bayonetta"
    bl_idname = "EXPORT_BN_main_menu"

    def draw(self, context):
        pcoll = preview_collections["main"]
        raiden_icon = pcoll["bayo"] 
        self.layout.operator(ExportBayoWMB.bl_idname, text="Model File (.wmb)", icon_value=raiden_icon.icon_id)

class B2BConfiguration(bpy.types.AddonPreferences):
    bl_idname = __package__
    astcEncDir:  bpy.props.StringProperty(name="ASTC Encoder Path", subtype='FILE_PATH')
    enableBayo: bpy.props.BoolProperty(name="Enable 'Bayonetta' Import Options")
    enableVanquish: bpy.props.BoolProperty(name="Enable 'Vanquish' Import Options")

    def draw(self, context):
        layout: bpy.types.UILayout = self.layout

        box = layout.box()
        box.label(text="Game Setup")


        box = layout.box()
        box.label(text="ASTC Encoder Setup")
        box.prop(self, "astcEncDir")

        op = box.operator("wm.url_open", text="Get ASTC Encoder", icon='URL')
        op.url = "https://github.com/ARM-software/astc-encoder/releases/"



classes = (
    BayonettaParameter,
    BayonettaTexture,
    Bayonetta2Data,
    BayonettaExMaterialData,
    BayoMaterialDataProperty,
    ImportBayoWMB,
    ImportNierDat,
    ExportBayoWMB,
    BayoMaterialPanel,
    BayoMaterialPanelAdvanced,
    BayoMaterialToJSON,
    BayoJSONToMaterial,
    BayonettaObjectMenu,
    RipMeshByUVIslands,
    IMPORT_BN_MainMenu,
    EXPORT_BN_MainMenu,
    BayoObjectPanel,
    ImportBayoSCR,
    B2BConfiguration

)






def menu_func_utils(self, context):
    pcoll = preview_collections["main"]
    raiden_icon = pcoll["bayo"]
    self.layout.menu(BayonettaObjectMenu.bl_idname, icon_value=raiden_icon.icon_id)

def menu_func_import(self, context):
    #self.layout.operator(ImportBayoWMB.bl_idname, text="Bayonetta WMB (.wmb)")
    #self.layout.operator(ImportNierDat.bl_idname, text="Bayonetta DAT (.dat)")
    pcoll = preview_collections["main"]
    raiden_icon = pcoll["bayo"] 
    
    self.layout.menu(IMPORT_BN_MainMenu.bl_idname, icon_value=raiden_icon.icon_id)

def menu_func_export(self, context):
    pcoll = preview_collections["main"]
    raiden_icon = pcoll["bayo"] 
    
    self.layout.menu(EXPORT_BN_MainMenu.bl_idname, icon_value=raiden_icon.icon_id)
    
    self.layout.operator_context = 'INVOKE_DEFAULT'
    #self.layout.operator(ExportBayoWMB.bl_idname, text="Bayonetta WMB (.wmb)")

def register():
    import bpy.utils.previews
    pcoll = bpy.utils.previews.new()
    my_icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    pcoll.load("bayo", os.path.join(my_icons_dir, "bayo.png"), 'IMAGE')
    pcoll.load("vanq", os.path.join(my_icons_dir, "vanq.png"), 'IMAGE')
    preview_collections["main"] = pcoll

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.VIEW3D_MT_object.append(menu_func_utils)
    bpy.types.Material.bayo_data = bpy.props.PointerProperty(type=BayoMaterialDataProperty)


    addon_dir = os.path.dirname(os.path.abspath(__file__))
    if (not os.path.exists(os.path.join(addon_dir, "userpref.json"))):
        print("Making userpref.json...")
        import json
        with open(os.path.join(addon_dir, "userpref.json"), "wt") as f:
            param = {}
            param["astcEnc"] = ""
            f.write(json.dumps(param))

def unregister():
    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()


    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    del bpy.types.Material.bayo_data

if __name__ == "__main__":
    register()