bl_info = {
    "name": "Bayonetta2Blender2Bayonetta",
    "author": "Gaming With Portals, Raq (With some things based off MGR2Blender2MGR)",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "description": "Import/Export Various Bayonetta Data files.",
    "category": "Import-Export"}


import bpy
from bpy.app.handlers import persistent
from .wmb.wmbImportOperator import ImportBayoWMB
from .wmb.wmbImportOperator import ExportBayoWMB
from .dat_dtt.importer.datImportOperator import ImportNierDat

classes = (
    ImportBayoWMB,
    ImportNierDat,
    ExportBayoWMB
)


def menu_func_import(self, context):
    self.layout.operator(ImportBayoWMB.bl_idname, text="Bayonetta WMB (.wmb)")
    self.layout.operator(ImportNierDat.bl_idname, text="Bayonetta DAT (.dat)")

def menu_func_export(self, context):
    self.layout.operator(ExportBayoWMB.bl_idname, text="Bayonetta WMB (.wmb)")

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()