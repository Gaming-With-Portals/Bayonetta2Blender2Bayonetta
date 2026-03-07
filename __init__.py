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
from .wmb.wmbImportOperator import ImportBayoWMB, ImportBayo2WMB
from .wmb.wmbImportOperator import ExportBayoWMB
from .dat_dtt.importer.datImportOperator import ImportNierDat
from .ui.material_ui import BayoMaterialPanel, BayoMaterialToJSON, BayoJSONToMaterial
from .ui.material_ui import BayoMaterialPanelAdvanced
from .utils.util import BayonettaVector4Property
from .wmb.wmb_materials import BayonettaParameter, BayonettaTexture
from .wmb.wmb_materials import BayoMaterialDataProperty
from .utils.utilOperators import RipMeshByUVIslands

class BayonettaObjectMenu(bpy.types.Menu):
    bl_idname = 'OBJECT_MT_b2b2b'
    bl_label = 'Bayonetta Tools'
    def draw(self, context):
        
        self.layout.operator(RipMeshByUVIslands.bl_idname, icon="UV_ISLANDSEL")

preview_collections = {}

class IMPORT_BN_MainMenu(bpy.types.Menu):
    bl_label = "Bayonetta"
    bl_idname = "IMPORT_MT_main_menu"

    def draw(self, context):
        pcoll = preview_collections["main"]
        raiden_icon = pcoll["bayo"] 
   
        self.layout.operator(ImportNierDat.bl_idname, text="Archive File (.dat, .dtt)", icon_value=raiden_icon.icon_id)
        self.layout.operator(ImportBayoWMB.bl_idname, text="Bayonetta 1 (.wmb)", icon_value=raiden_icon.icon_id)
        self.layout.operator(ImportBayo2WMB.bl_idname, text="Bayonetta 2 (.wmb)", icon_value=raiden_icon.icon_id)

class EXPORT_BN_MainMenu(bpy.types.Menu):
    bl_label = "Bayonetta"
    bl_idname = "EXPORT_MT_main_menu"

    def draw(self, context):
        pcoll = preview_collections["main"]
        raiden_icon = pcoll["bayo"] 
        self.layout.operator(ExportBayoWMB.bl_idname, text="Model File (.wmb)", icon_value=raiden_icon.icon_id)


classes = (
    BayonettaParameter,
    BayonettaTexture,
    BayoMaterialDataProperty,
    ImportBayoWMB,
    ImportBayo2WMB,
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
    preview_collections["main"] = pcoll

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.VIEW3D_MT_object.append(menu_func_utils)
    bpy.types.Material.bayo_data = bpy.props.PointerProperty(type=BayoMaterialDataProperty)

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