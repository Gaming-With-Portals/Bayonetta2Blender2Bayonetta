import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper


class ImportBayoWMB(bpy.types.Operator, ExportHelper):
    '''Import WMB Data.'''
    bl_idname = "import_scene.bayo_wmb_data"
    bl_label = "Import WMB Data"
    bl_options = {'PRESET'}
    filename_ext = ".wmb"
    filter_glob: StringProperty(default="*.wmb", options={'HIDDEN'})

    #reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)
    bone_names: bpy.props.BoolProperty(name="Use Custom Bone Names", default=True)
    shadow_meshes: bpy.props.BoolProperty(name="Hide Shadow Meshes", default=True)

    def execute(self, context):
        from . import wmb_importer
        return  wmb_importer.ImportWMB(self.filepath, "", self.bone_names, self.shadow_meshes)
    
class ImportBayo2WMB(bpy.types.Operator, ExportHelper):
    '''Import WMB Data.'''
    bl_idname = "import_scene.bayo2_wmb_data"
    bl_label = "Import WMB Data"
    bl_options = {'PRESET'}
    filename_ext = ".wmb"
    filter_glob: StringProperty(default="*.wmb", options={'HIDDEN'})

    #reset_blend: bpy.props.BoolProperty(name="Reset Blender Scene on Import", default=True)
    bone_names: bpy.props.BoolProperty(name="Use Custom Bone Names", default=True)
    shadow_meshes: bpy.props.BoolProperty(name="Hide Shadow Meshes", default=True)
    normal_type: bpy.props.EnumProperty(
        name="Normal Version",
        description="Select the type of normals to use during import",
        items=[
            ('PC',    "PC",             "Import using Bayonetta 1 PC Normals"),
            ('SWITCH',  "Nintendo Switch",   "Import using Bayonetta 2 Nintendo Switch Normals"),
            ('WIIU',    "Wii U",     "Import using Bayonetta 2 Nintendo Wii U Normals")
        ],
        default='SWITCH'
    )

    def execute(self, context):
        from . import wmb_importer
        return  wmb_importer.ImportWMB(self.filepath, "", self.bone_names, self.shadow_meshes, True, self.normal_type)
    
class ExportBayoWMB(bpy.types.Operator, ExportHelper):
    '''Export WMB Data.'''
    bl_idname = "export.bayo_wmb_data"
    bl_label = "Export WMB File"
    bl_options = {'PRESET'}
    filename_ext = ".wmb"
    filter_glob: StringProperty(default="*.wmb", options={'HIDDEN'})

    btt: bpy.props.BoolProperty(name="Generate Bone Index Translate Table", default=True)
    large_bone: bpy.props.BoolProperty(name="Use Skyth's Large Bone Patch", default=False)
    copy_uv: bpy.props.BoolProperty(name="Use UVMap1 as UVMap2", default=True)

    def execute(self, context):
        from . import wmb_exporter
        return  wmb_exporter.export(self.filepath, self, False, self.btt, self.large_bone, self.copy_uv)

