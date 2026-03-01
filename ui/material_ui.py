import bpy
from ..utils.util import BayonettaVector4Property
import json

class BayoMaterialToJSON(bpy.types.Operator):
    bl_idname = "b2b.materialtojson"
    bl_label = "Copy material to JSON"

    def execute(self, context):
        material = bpy.context.material

        data_structure = {}
        data_structure["version"]=1
        data_structure["is_b2"]=material.bayo_data.bayonetta_2
        data_structure["type"]=material.bayo_data.type
        data_structure["flags"]=material.bayo_data.flags

        data_structure["parameters"] = []
        for param in material.bayo_data.parameters:
            serialized_param = {}
            serialized_param["type"] = param.type
            serialized_param["name"] = param.name
            serialized_param["data"] = {
                "int": param.value_int,
                "float": param.value_float,
                "vec2": list(param.value_vec2) if hasattr(param.value_vec2, "__iter__") else param.value_vec2,
                "vec3": list(param.value_vec3) if hasattr(param.value_vec3, "__iter__") else param.value_vec3,
                "vec4": list(param.value_vec4) if hasattr(param.value_vec4, "__iter__") else param.value_vec4
            }
            data_structure["parameters"].append(serialized_param)

        material["wmb_mat_as_json"] = json.dumps(data_structure)
        bpy.context.window_manager.clipboard = bpy.context.material["wmb_mat_as_json"]
        self.report({'INFO'}, "Copied JSON")
        return {'FINISHED'}

class BayoJSONToMaterial(bpy.types.Operator):
    bl_idname = "b2b.jsontomaterial"
    bl_label = "Paste JSON into material"

    def execute(self, context):
        material = bpy.context.material
        bpy.context.material["wmb_mat_as_json"] = bpy.context.window_manager.clipboard
        try:
            data_structure = json.loads(material["wmb_mat_as_json"])
        except:
            self.report({'ERROR'}, "Invalid JSON")
            return {'FINISHED'}


        if (data_structure["version"] != 1):
            self.report({'ERROR'}, "Unsupported JSON Format (likely too new, update your plugin)")

        if not data_structure["is_b2"]:
            material.bayo_data.bayonetta_2 = False
            material.bayo_data.type = data_structure["type"]
            material.bayo_data.flags = data_structure["flags"]
            material.bayo_data.parameters.clear()

            i=0
            for param in data_structure["parameters"]:
                material.bayo_data.parameters.add()
                material.bayo_data.parameters[i].type = param["type"]
                material.bayo_data.parameters[i].name = param["name"]

                if (param["type"] == "sampler2D_t" or param["type"] == "samplerCUBE_t"):
                    material.bayo_data.parameters[i].value_int = param["data"]["int"]
                elif param["type"] == "f4_float3_t":
                    material.bayo_data.parameters[i].value_vec3 = param["data"]["vec3"]
                elif param["type"] == "f4_float2_t":
                    material.bayo_data.parameters[i].value_vec2 = param["data"]["vec2"]
                elif param["type"] == "f4_float_t":
                    material.bayo_data.parameters[i].value_float = param["data"]["float"]
                else:
                    material.bayo_data.parameters[i].value_vec4 = param["data"]["vec4"]

                i+=1
            


        self.report({'INFO'}, "Pasted JSON")
        return {'FINISHED'}



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
    
        layout.operator(BayoMaterialToJSON.bl_idname, text="Copy Material")
        layout.operator(BayoJSONToMaterial.bl_idname, text="Paste Material")
        
