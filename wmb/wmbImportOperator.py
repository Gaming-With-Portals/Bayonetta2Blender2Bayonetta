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
        f = open(self.filepath, "rb")
        tag = f.read(4)
        if (tag == b"WMB\x00" or tag == b"\x00BMW"):
            from .wmb0 import wmb_importer
            return  wmb_importer.ImportWMB(self.filepath, "", self.bone_names, self.shadow_meshes)
        elif (tag == b"WMB3"):
            from .wmb3 import wmb3_importer
            return wmb3_importer.ImportWMB3(self.filepath)
        else:
            print(f"[!] Unsupport WMB version: {tag.decode()}")
            return {"CANCELLED"}




    
class ExportBayoWMB(bpy.types.Operator, ExportHelper):
    '''Export WMB Data.'''
    bl_idname = "export.bayo_wmb_data"
    bl_label = "Export WMB File"
    bl_options = {'PRESET'}
    filename_ext = ".wmb"
    filter_glob: StringProperty(default="*.wmb", options={'HIDDEN'})

    game_name: bpy.props.EnumProperty(
        name="Game",
        description="The game target to cater the exporter towards",
        items=[
            ('AUTO', "Automatic", "Choose the version based on what you imported"),
            ('BAYO1', "Bayonetta 1", "Bayonetta 1 (2009)"),
            ('VANQ', "Vanquish", "Vanquish (2010)"),
            ('W101', "The Wonderful 101", "The Wonderful 101 (2013)"),
            ('BAYO2', "Bayonetta 2", "Bayonetta 2 (2014)")
        ],
        default='AUTO'
    )
    platform: bpy.props.EnumProperty(
        name="Platform",
        description="The platform target to cater the exporter towards",
        items=[
            ('PC', "PC", "PC Version (Windows, Mac, Linux)"),
            ("SWITCH", "Nintendo Switch", "Nintendo Switch/Switch 2 Versions"),
            ("WIIU", "Wii U", "Wii U Versions"),
            ("X360", "Xbox 360", "Xbox 360 Versions"),
            ("PS3", "Playstation 3", "Playstation 3 Versions")
        ],
        default='PC'
    )

    #btt: bpy.props.BoolProperty(name="Generate Bone Index Translate Table", default=True)
    large_bone: bpy.props.BoolProperty(name="Use Skyth's Large Bone Patch", default=False)
    #copy_uv: bpy.props.BoolProperty(name="Use UVMap1 as UVMap2", default=True)

    def invoke(self, context, event):
        if ("WMB" in bpy.context.view_layer.layer_collection.children):
            wmb_collection = bpy.context.view_layer.layer_collection.children["WMB"]
            sub_collection = [x for x in wmb_collection.children if x.is_visible][0]
            arm_obj = sub_collection.collection.objects[0]

            if arm_obj.get("large_bones"):
                self.large_bone = True

        return super().invoke(context, event)


    def execute(self, context):
        from .wmb0 import wmb_exporter
        return  wmb_exporter.export(self.filepath, self, False, True, self.large_bone, False, platform=self.platform, gamename=self.game_name)

