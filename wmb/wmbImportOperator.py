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



    #btt: bpy.props.BoolProperty(name="Generate Bone Index Translate Table", default=True)
    #large_bone: bpy.props.BoolProperty(name="Use Skyth's Large Bone Patch", default=False)
    #keep_refs: bpy.props.BoolProperty(name="Keep Original Bone Refs", default=False)
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
        from .wmb3 import wmb3_exporter
        return  wmb3_exporter.export(self.filepath)

