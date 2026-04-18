import bpy
import struct
import json
from io import BufferedReader
import os
import math
from mathutils import Vector
import bmesh
from .wmb_materials import materialSizeDictionary
from .wmb_bone_names import getBoneName
from ..structwrapper import BinReader

wmb_material_list = {}
wmb_texture_list = {}

def invertGreenChannel(nodes, normal_node_pos=0):
    # Thanks Naq, you're a king fr
    invert_node = nodes.new("ShaderNodeRGBCurve")
    invert_node.name = "invert"
    invert_node.label = "Invert Green Channel"
    invert_node.location = 450, normal_node_pos
    invert_node.hide = True
    #node.label = texture_nrm.name

    # Let's invert the green channel
    green_channel = invert_node.mapping.curves[1] # Second curve is for green
    green_channel.points[0].location.y = 1
    green_channel.points[1].location.y = 0
    blue_channel = invert_node.mapping.curves[1] 
    blue_channel.points[0].location.y = 1
    blue_channel.points[1].location.y = 1    
    
    return invert_node

def decode_triangle_strip(strip_indices):
    triangles = []
    for i in range(len(strip_indices) - 2):
        i0, i1, i2 = strip_indices[i], strip_indices[i+1], strip_indices[i+2]

        # Skip degenerate triangles
        if i0 == i1 or i1 == i2 or i2 == i0:
            continue

        # Even or odd triangle index determines winding
        if i % 2 == 0:
            triangles.append((i0, i1, i2))
        else:
            triangles.append((i1, i0, i2))
    return triangles

class WMBVector:
    x = 0
    y = 0 
    z = 0
    def __init__(self, file):
        self.x = struct.unpack("<f", file.read_float32())
        self.y = struct.unpack("<f", file.read_float32())
        self.z = struct.unpack("<f", file.read_float32())

class WMBBatchHeader:
    batchIdx = 0
    id = 0
    flags = 0
    exMaterialID = 0
    materialID = 0
    hasBoneRefs = 0
    unknownE1 = 0
    unknownE2 = 0
    vertexStart = 0
    vertexEnd = 0
    primativeType = 0
    offsetIndices = 0
    numIndices = 0
    vertexOffset = 0

    def __init__(self, file : BinReader):
        self.batchIdx = file.read_s16()
        self.id = file.read_s16()
        self.flags = file.read_u16()
        self.exMaterialID = file.read_s16()
        self.materialID = file.read_u8()
        self.hasBoneRefs = file.read_u8()
        self.unknownE1 = file.read_u8()
        self.unknownE2 = file.read_u8()
        self.vertexStart = file.read_u32()
        self.vertexEnd = file.read_u32()
        self.primativeType = file.read_u32()
        self.offsetIndices = file.read_u32()
        self.numIndices = file.read_u32()
        self.vertexOffset = file.read_u32()

class WMBMesh:
    id = 0
    batchCount = 0
    unknownA1 = 0
    boundingBoxInfos = 0
    offsetBatchOffsets = 0
    flags = 0
    name = ""
    center = None
    height = 0
    corner1 = None
    corner2 = None
    unknownD = 0
    unknownE = 0

    def __init__(self, file : BinReader):
        self.id = file.read_s16()
        self.batchCount = file.read_s16()
        self.unknownA1 = file.read_s16()
        self.boundingBoxInfos = file.read_s16()
        self.offsetBatchOffsets = file.read_u32()
        self.flags = file.read_s32()
        file.read(16)
        self.name = file.read(32).decode()
        self.center = WMBVector(file)
        self.height = file.read_float32()
        self.corner1 = WMBVector(file)
        self.corner2 = WMBVector(file)  
        self.unknownD = file.read_float32()
        self.unknownE = file.read_float32()         

class WMBMaterial:
    matID = 0
    flags = 0
    data = 0
    sampler_1_id = 0
    sampler_2_id = 0

    bpyMaterial = None
    def __init__(self, file : BinReader, json=None, texinfojson=None):
        self.parameter_data = {}
        self.matID = file.read_s16()
        self.flags = file.read_s16()
        datasize = materialSizeDictionary[self.matID] - 4
        data_start = file.tell()
        
        self.texinfo = texinfojson


        if json is not None:
            if str(self.matID) in json:
                mat_info = json[str(self.matID)]
                self.layout = mat_info.get(":layout", {})
                file.seek(data_start)
                for param_name, param_type in self.layout.items():
                    if param_type == "sampler2D_t" or param_type == "samplerCUBE_t":
                        self.parameter_data[param_name] = file.read_s32()
                    elif param_type == "f4_float3_t":
                        self.parameter_data[param_name] = (struct.unpack(f"{file.end_flag}fff", file.read(12)))
                        file.read(4)
                    elif param_type == "f4_float2_t":
                        self.parameter_data[param_name] = (struct.unpack(f"{file.end_flag}ff", file.read(8)))
                        file.read(8)
                    elif param_type == "f4_float_t":
                        self.parameter_data[param_name] = file.read_float32()
                        file.read(12)
                    else:
                        self.parameter_data[param_name] = (struct.unpack(f"{file.end_flag}ffff", file.read(16)))

                    

            else:
                print("Material type isn't in the JSON... it may be invalid.")
        file.seek(data_start)
        self.data = struct.unpack(file.end_flag + str(datasize // 4) + "i", file.read(materialSizeDictionary[self.matID] - 4))
        self.sampler_1_id = self.data[0]
        self.sampler_2_id = self.data[1]


    def toBPYMaterial(self, material_name, texture_path=""):
        global wmb_material_list
        global wmb_texture_list
        if material_name in wmb_material_list:
            return wmb_material_list[material_name]



        mat = bpy.data.materials.new(material_name)
        mat.use_nodes = True
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
        mat.bayo_data.parameters.clear()
        mat.bayo_data.type = self.matID
        mat.bayo_data.flags = self.flags
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        i = 0
        for param_name, param_type in self.layout.items():
            mat.bayo_data.parameters.add()
            mat.bayo_data.parameters[i].type = param_type
            mat.bayo_data.parameters[i].name = param_name
            if param_type == "sampler2D_t" or param_type == "samplerCUBE_t":
                mat.bayo_data.parameters[i].value_int = self.parameter_data[param_name]
            elif param_type == "f4_float3_t":
                mat.bayo_data.parameters[i].value_vec3 = self.parameter_data[param_name]
            elif param_type == "f4_float2_t":
                mat.bayo_data.parameters[i].value_vec2 = self.parameter_data[param_name]
            elif param_type == "f4_float_t":
                mat.bayo_data.parameters[i].value_float = self.parameter_data[param_name]
            else:
                mat.bayo_data.parameters[i].value_vec4 = self.parameter_data[param_name]


            i+=1


        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = 1300,0
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = 1000,0
        output_link = links.new( principled.outputs['BSDF'], output.inputs['Surface'] )

        albedo_bpy_image = None
        if self.sampler_1_id in wmb_texture_list:
            print("Texture already exists, no need to create another")
            albedo_bpy_image = wmb_texture_list[self.sampler_1_id]

        else:
            print(os.path.join(texture_path, f"{self.sampler_1_id:0>8X}.dds"))
            if (os.path.isfile(os.path.join(texture_path, f"{self.sampler_1_id:0>8X}.dds"))):
                albedo_bpy_image = bpy.data.images.load(os.path.join(texture_path, f"{self.sampler_1_id:0>8X}.dds"))

        normal_bpy_image = None
        if self.sampler_2_id in wmb_texture_list:
            print("Texture already exists, no need to create another")
            normal_bpy_image = wmb_texture_list[self.sampler_2_id]

        else:
            print(os.path.join(texture_path, f"{self.sampler_2_id:0>8X}.dds"))
            if (os.path.isfile(os.path.join(texture_path, f"{self.sampler_2_id:0>8X}.dds"))):
                normal_bpy_image = bpy.data.images.load(os.path.join(texture_path, f"{self.sampler_2_id:0>8X}.dds"))


        alb_image_node = nodes.new(type='ShaderNodeTexImage')
        alb_image_node.location = -100,-60
        if albedo_bpy_image is not None:
            alb_image_node.image = albedo_bpy_image
            if ("flags" in self.texinfo):
                print("flag")
                if (self.matID in self.texinfo["flags"]):
                    flag = self.texinfo["flags"][self.matID]
                    if (flag == 268435456):
                        links.new(alb_image_node.outputs['Alpha'], principled.inputs["Alpha"])


        nrm_image_node = nodes.new(type='ShaderNodeTexImage')
        nrm_image_node.location = -100,-500
        if normal_bpy_image is not None:
            nrm_image_node.image = normal_bpy_image
            normalShader = nodes.new(type='ShaderNodeNormalMap')
            normalShader.location = 0, 0
            normalShader.hide = True
            invertGreen = invertGreenChannel(nodes, nrm_image_node.location[1])
            links.new(nrm_image_node.outputs["Color"], invertGreen.inputs["Color"])
            links.new(invertGreen.outputs["Color"], normalShader.inputs["Color"])
            #links.new(normalShader.outputs["Normal"], principled.inputs["Normal"])
            




        links.new(alb_image_node.outputs['Color'], principled.inputs["Base Color"])
        wmb_material_list[material_name] = mat




        mat["type"] = self.matID
        mat["flags"] = self.flags
        mat["data"] = self.data
        mat["size"] = materialSizeDictionary[self.matID]
        
        return mat
    
class WMBMaterial2:
    

    def __init__(self, file, shader_name, size, ids, tex_ids_to_type):
        self.matID = struct.unpack("<h", file.read(2))[0]
        self.flags = struct.unpack("<h", file.read(2))[0]
        self.texture_data = []
        self.data_data = []
        self.shader_name = shader_name
        self.tex_id_list = ids
        self.id_type_map = tex_ids_to_type
        print(f"{size} - {(size - 24) // 4}")
        for i in range(5):
            self.texture_data.append(struct.unpack("<i", file.read(4))[0])
        for i in range((size - 24) // 4):
            self.data_data.append(struct.unpack("<f", file.read(4))[0])


    def toBPYMaterial(self, material_name, texture_path=""):
        global wmb_material_list
        global wmb_texture_list
        if material_name in wmb_material_list:
            return wmb_material_list[material_name]

        mat = bpy.data.materials.new(material_name)
        mat.bayo_data.parameters.clear()
        mat.bayo_data.type = self.matID
        mat.bayo_data.flags = self.flags
        mat.bayo_data.bayonetta_2 = True
        mat.use_nodes = True
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

        i = 0
        for tex in self.texture_data:
            mat.bayo_data.textures.add()
            mat.bayo_data.textures[i].id = self.texture_data[i]

            i+=1

        mat["id"] = self.matID
        mat["flags"] = self.flags

        self.texture_ids = []
        self.texture_datas = []
        for tex in self.texture_data:
            if (tex in self.tex_id_list):
                self.texture_ids.append(tex)
            else:
                self.texture_datas.append(tex)

        mat["texture_ids"] = self.texture_ids
        mat["texture_data"] = self.texture_datas
        mat["raw_data"] = self.texture_data
        mat["data"] = self.data_data
        mat["shader"] = self.shader_name

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = 1300,0
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.inputs["Roughness"].default_value = 1
        principled.location = 1000,0
        output_link = links.new( principled.outputs['BSDF'], output.inputs['Surface'] )

        normal_bpy_image = None
        albedo_bpy_image = None
        for i, tex in enumerate(self.texture_data):
            if (tex in self.id_type_map):
                tp = self.id_type_map[tex]

       
                if tex in wmb_texture_list:
                    if (i == 0):
                        albedo_bpy_image = wmb_texture_list[tex]
                    if (tp == 15):
                        normal_bpy_image = wmb_texture_list[tex]

                else:
                    
                    if (os.path.isfile(os.path.join(texture_path, f"{tex:0>8X}.dds"))):
                        if (i == 0):
                            albedo_bpy_image = bpy.data.images.load(os.path.join(texture_path, f"{tex:0>8X}.dds"))
                            if (albedo_bpy_image == None):
                                albedo_bpy_image = bpy.data.images.load(os.path.join(texture_path, f"{tex:0>8X}.png"))
                        if (tp == 15):
                            normal_bpy_image = bpy.data.images.load(os.path.join(texture_path, f"{tex:0>8X}.dds"))
                            if (normal_bpy_image == None):
                                normal_bpy_image = bpy.data.images.load(os.path.join(texture_path, f"{tex:0>8X}.png"))



        alb_image_node = nodes.new(type='ShaderNodeTexImage')
        alb_image_node.location = -100,-60
        if albedo_bpy_image is not None:
            alb_image_node.image = albedo_bpy_image
            links.new(alb_image_node.outputs['Color'], principled.inputs["Base Color"])

        nrm_image_node = nodes.new(type='ShaderNodeTexImage')
        nrm_image_node.location = -100,-500
        if normal_bpy_image is not None:
            nrm_image_node.image = normal_bpy_image
            normalShader = nodes.new(type='ShaderNodeNormalMap')
            normalShader.location = 0, 0
            normalShader.hide = True
            links.new(nrm_image_node.outputs["Color"], normalShader.inputs["Color"])
            links.new(normalShader.outputs["Normal"], principled.inputs["Normal"])

        

        wmb_material_list[material_name] = mat
        return mat

def decode_bayo_switch_normal(r):
    '''nx = packed_value & ((1 << 10) - 1)
    ny = (packed_value >> 10) & ((1 << 10) - 1)
    nz = (packed_value >> 20) & ((1 << 10) - 1)
    sign = nx & (1<<9)
    if (sign):
        nx ^= sign
        nx = -(sign-nx)
    
    sign = ny & (1<<9)
    if (sign):
        ny ^= sign
        ny = -(sign-ny)

    sign = nz & (1<<9)
    if (sign):
        nz ^= sign
        nz = -(sign-nz)

    fx = nx/((1<<9)-1)
    fy = ny/((1<<9)-1)
    fz = nz/((1<<9)-1)

    return fx, fy, fz'''

    def sign_extend(value: int, bits: int) -> int:
        sign_bit = 1 << (bits - 1)
        if value & sign_bit:
            value -= (1 << bits)
        return value

    x = (r >> 0)  & 0x3FF  # bits 0-9
    y = (r >> 10) & 0x3FF  # bits 10-19
    z = (r >> 20) & 0x3FF  # bits 20-29

    scale = (1 << 9) - 1  # 511.0

    return (
        sign_extend(x, 10) / scale,
        sign_extend(y, 10) / scale,
        sign_extend(z, 10) / scale,
    )

def decode_bayonetta_normals(f : BinReader, nType):
    bpyVector = Vector((0, 0, 0))

    if (nType == "B1"):
        f.advance(1)  # dummy byte[0]
        nz, ny, nx = f.read_s8_array(3)  # byte[1], byte[2], byte[3]

        bpyVector = Vector((nx / 127.0, ny / 127.0, nz / 127.0))

        if bpyVector.length == 0:
            bpyVector = Vector((0.0, 0.0, 1.0))
        else:
            bpyVector.normalize()
    elif (nType == "B2"):
        bayoPack = f.read_u32()

        nx = bayoPack & ((1 << 10) - 1)
        ny = (bayoPack >> 10) & ((1 << 10) - 1)
        nz = (bayoPack >> 20) & ((1 << 10) - 1)
        sign = nx & (1<<9)
        if (sign):
            nx ^= sign
            nx = -(sign-nx)
            
        sign = ny & (1<<9)
        if (sign):
            ny ^= sign
            ny = -(sign-ny)

        sign = nz & (1<<9)
        if (sign):
            nz ^= sign
            nz = -(sign-nz)

        mag = (1<<9)-1
        fx = nx/mag
        fy = ny/mag
        fz = nz/mag

        bpyVector = Vector((fx, fy, fz))

        if bpyVector.length == 0:
            bpyVector = Vector((0.0, 0.0, 1.0))
        else:
            bpyVector.normalize()

    

    return bpyVector

    
def ImportWMB(filepath, textures, use_custom_bone_names, hide_shadow_meshes, bayo_2=False, normal_type="B1", target_col="WMB", cmn_texture_src=""):
    import numpy as np

    def read_half_float(hf_bytes):
        return float(np.frombuffer(hf_bytes, dtype=np.float16)[0])
    
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(addon_dir, "..", "materials.json")
    json_path = os.path.abspath(json_path)  # Normalize it
    material_json = None
    if os.path.isfile(json_path):
        with open(json_path, "rt", encoding="utf-8") as f:
            material_json = json.load(f)

    is_wii = False

    if (target_col=="WMB"):
        if ("WMB" not in bpy.data.collections):
            wmb_collection = bpy.data.collections.new("WMB")
            bpy.context.scene.collection.children.link(wmb_collection)
        else:
            wmb_collection = bpy.data.collections["WMB"]
    else:
        wmb_collection = bpy.data.collections[target_col] # Caller is responsible for creating the collection
        
    
    with open(filepath, 'rb') as f:
        wf = BinReader(f)
        if (wf.read(4).decode() == "\x00BMW"):
            print("Wii U/Xbox 360/PS3 (Big Endian) file!")
            wf.big = True
            is_wii = True
            normal_type = "B1_WII_U"
            wf.update_endianess_flag()

        wf.seek(8)
        vertexFormat = wf.read_u32()
        num_vertices =wf.read_u32()
        num_uvmaps = wf.read_u8()
        num_colors = wf.read_u8()
        wf.advance(2)
        offset_positions = wf.read_u32()
        offset_vertices = wf.read_u32()
        offset_vertices_extra = wf.read_u32()
        wf.advance(16)
        numBones = wf.read_u32()
        offsetBoneHierarchy = wf.read_u32()
        offsetBoneRelativePosition = wf.read_u32()
        offsetBonePositions = wf.read_u32()
        offsetBoneIndexTranslateTable = wf.read_u32()
        numMaterials = wf.read_u32()
        offsetMaterialsOffsets = wf.read_u32()
        offsetMaterials = wf.read_u32()
        numMeshes = wf.read_u32()
        offsetMeshesOffsets = wf.read_u32()
        offsetMeshes = wf.read_u32()
        numPolygons = wf.read_u32()
        numShaderSettings = wf.read_u32()
        offsetInverseKinematics = wf.read_u32()
        offsetBoneSymmetries = wf.read_u32()
        offsetBoneFlags = wf.read_u32()
        exMatShaderNamesOffset = wf.read_u32()
        exMatSamplersOffset = wf.read_u32()
        exMatInfo = (wf.read_u32(), wf.read_u32())

        if (exMatSamplersOffset != 0 and exMatShaderNamesOffset != 0):
            normal_type="B2"
            bayo_2 = True

        if (not bayo_2):
            if (vertexFormat != 0x4000001F):
                wf.seek(offset_vertices+0x10)
                if (wf.read_u8() != 0):
                    normal_type = "B1_NINTENDO_SWITCH"

        print(f"Normal type is {normal_type}")

        wmb_name = os.path.splitext(os.path.basename(filepath))[0]

        model_collection = bpy.data.collections.new(wmb_name)
        model_collection["vertex_format"] = vertexFormat

        wmb_collection.children.link(model_collection)

        shader_names = []
        if (exMatShaderNamesOffset != 0):
            wf.seek(exMatShaderNamesOffset)
            for _ in range(numMaterials):
                shader_names.append(wf.read(16).decode().replace("\x00", ""))

        tex_ids_to_type = {}
        tex_ids = []
        if (exMatSamplersOffset != 0):
            wf.seek(exMatSamplersOffset)
            texCount = wf.read_u32()
            for _ in range(texCount):
                tex_id = wf.read_u32()
                flag = wf.read_u32()
                tex_ids.append(tex_id)
                tex_ids_to_type[tex_id] = flag

        material_flag_info = {}
        if (os.path.exists(os.path.join(textures, "texinfo.json"))):
            material_flag_info = json.load(open(os.path.join(textures, "texinfo.json"), "rt"))

        wf.seek(offsetMaterialsOffsets)
        bayonettaMaterialList = []
        materialOffsets = wf.read_u32_array(numMaterials)
        for x in range(numMaterials):
            start = materialOffsets[x]

            if x == numMaterials - 1:
                size = offsetMeshesOffsets - (offsetMaterials + start)
            else:
                size = materialOffsets[x+1] - start

            wf.seek(offsetMaterials + start) # not confusing at all :fire:
            if (bayo_2 == False):
                materialData = WMBMaterial(wf, material_json, material_flag_info) # TODO: Use WF System
            else:
                materialData = WMBMaterial2(wf, shader_names[x], size, tex_ids, tex_ids_to_type) # TODO: Use WF System

            bayonettaMaterialList.append(materialData)


        flag_map = {}
        if (offsetBoneFlags != 0):
            wf.seek(offsetBoneFlags)
            for i in range(numBones):
                flag_map[i] = wf.read_u8()

        # build skel
        if numBones > 0:
            arm_data = bpy.data.armatures.new(wmb_name)
            arm_obj = bpy.data.objects.new(wmb_name, arm_data)
            model_collection.objects.link(arm_obj)
            bpy.context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='EDIT')

            # bone hierarchy
            wf.seek(offsetBoneHierarchy)
            bone_parents = [wf.read_s16() for _ in range(numBones)]

            arm_obj["bone_hierarchy"] = bone_parents # this really shouldn't be that bad

            # absolute positions
            wf.seek(offsetBonePositions)
            bone_abs_positions = [Vector(wf.read_f32_vector3()) for _ in range(numBones)]

            # Bullshit bone remap thing, check later
            bone_name_map = {}
            local_bone_name_map = {}
            bone_id_map = {}
            if offsetBoneIndexTranslateTable:
                parts_map = []
                wf.seek(offsetBoneIndexTranslateTable)

                l1_table = [wf.read_u16() for _ in range(16)]

                for l1_index in range(16):
                    l2_offset = l1_table[l1_index]
                    if l2_offset == 0xFFFF:
                        continue

                    for l2_index in range(16):
                        wf.seek(offsetBoneIndexTranslateTable + ((l2_offset + l2_index) * 2))
                        l3_offset = wf.read_u16()
                        if l3_offset == 0xFFFF:
                            continue

                        for l3_index in range(16):
                            wf.seek(offsetBoneIndexTranslateTable + ((l3_offset + l3_index) * 2))
                            parts_index = wf.read_u16()
                            if parts_index != 0xFFF:
                                parts_no = (l1_index << 8) | (l2_index << 4) | l3_index
                                parts_map.append((parts_no, parts_index))                            

                for item in parts_map:
                    #bone_name_map[item[1]] = f"bone{item[1]:03}"
                    bone_name_map[item[1]] = getBoneName(item[1], item[0], use_custom_bone_names)
                    bone_id_map[item[1]] = item[0]


                wf.seek(offsetBoneIndexTranslateTable) # TODO, nuke from orbit
                # first level
                first_level = [wf.read_s16() for _ in range(16)]
                j = sum(1 for val in first_level if val != -1 and val != 4095)
                arm_obj["translate_table_1"] = first_level
                # second level
                second_level = [wf.read_s16() for _ in range(j * 16)]
                k = sum(1 for val in second_level if val != -1 and val != 4095)
                arm_obj["translate_table_2"] = second_level
                # third level
                third_level = [wf.read_s16() for _ in range(k * 16)]
                arm_obj["translate_table_3"] = third_level
                arm_obj["translate_table_size"] = f.tell() - offsetBoneIndexTranslateTable


            edit_bones = {}
            for i in range(numBones):
                bone_name = bone_name_map.get(i, f"bone{i:04}")
                bone_id = bone_id_map.get(i, -1)
                bone = arm_data.edit_bones.new(bone_name)
                bone["local_id"] = i
                raw = bone_abs_positions[i]
                converted = Vector((raw.x, -raw.z, raw.y))
                bone.head = converted
                bone.tail = bone.head + Vector((0.0, 0.05, 0.0))
                if (offsetBoneFlags != 0):
                    bone["flags"] = flag_map[i]
                bone["id"] = bone_id
                edit_bones[i] = bone

            for i, parent_index in enumerate(bone_parents):
                if parent_index != -1:
                    edit_bones[i].parent = edit_bones[parent_index]
                    edit_bones[i].use_connect = False

            bpy.ops.object.mode_set(mode='OBJECT')

            arm_obj.show_in_front = True
            arm_obj.data.display_type = 'STICK'

            def bone_group_name(bone_id):
                return bone_name_map.get(bone_id, f"bone{bone_id:03}")

            arm_obj["bone_flags"] = False
            if (offsetBoneFlags != 0):
                arm_obj["bone_flags"] = True

            arm_obj["bone_symmetries"] = False
            if (offsetBoneSymmetries != 0):
                arm_obj["bone_symmetries"] = True

            arm_obj["inverse_kinematics"] = False
            if (offsetInverseKinematics != 0):
                arm_obj["inverse_kinematics"] = True
                wf.seek(offsetInverseKinematics)
                kincount = wf.read_s8()
                other_data = wf.read_s8_array(3)
                offset = wf.read_s32()
                arm_obj["ik_count"] = kincount
                arm_obj["ik_offset"] = offset
                arm_obj["ik_unk"] = other_data
                wf.seek(offsetInverseKinematics + offset)
                for i in range(kincount):
                    structure = wf.read_s8_array(16)
                    arm_obj["ik_structure_" + str(i)] = structure

            arm_obj["b2"] = bayo_2
            if (bayo_2):
                arm_obj["exmatinfo"] = exMatInfo

            for id, flag in tex_ids_to_type.items():
                arm_obj[f"txtr{id}"] = flag

        else:
            model_collection["b2"] = bayo_2
            if (bayo_2):
                model_collection["exmatinfo"] = exMatInfo

            for id, flag in tex_ids_to_type.items():
                model_collection[f"txtr{id}"] = flag


        # Read vertex positions separately if offset_positions is non-zero
        vertices, vcolor, uvs, normals, vertex_groups_data = [], [], [], [], []
        extra_vcolors = []
        uv2s = []

        if vertexFormat == 0x4000001F:
            # read positions from offset_positions
            wf.seek(offset_positions)
            positions = [wf.read_f32_vector3() for _ in range(num_vertices)]

            # now read the rest of the vertex data
            wf.seek(offset_vertices)

            for i in range(num_vertices):
                px, py, pz = positions[i]

                # Normal as 3 floats
                nx, ny, nz = wf.read_f32_vector3()
                normal = Vector((nx, ny, nz))
                if normal.length == 0:
                    normal = Vector((0.0, 0.0, 1.0))
                else:
                    normal.normalize()

                r, g, b, a = wf.read_u8_array(4)
                vcolor.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))
                wf.advance(4)  # tangents - skip

                bone_ids = wf.read_u8_array(4)
                bone_weights_raw = wf.read_u8_array(4)

                u, v = wf.read_float32(), wf.read_float32()  # UVs as full floats

                weights_sum = sum(bone_weights_raw)
                if weights_sum == 0:
                    weights_sum = 1
                bone_weights = [w / weights_sum for w in bone_weights_raw]

                vertices.append((px, py, pz))
                uvs.append((u, 1.0 - v))
                normals.append(normal)
                vertex_groups_data.append(list(zip(bone_ids, bone_weights)))

        elif (vertexFormat == 0x5800000F or vertexFormat == 0x4B40000F):
            # SCR Vertex format (Position, UV, Normals, Tangents, Color UV2 [if uvCount=2])
            wf.seek(offset_vertices)
            for i in range(num_vertices):
                px, py, pz = wf.read_f32_vector3()

                u = read_half_float(wf.read(2))
                v = read_half_float(wf.read(2))
                normal = decode_bayonetta_normals(wf, normal_type)
                '''if (normal_type == "B1"):
                    wf.advance(1)
                    nz, ny, nx = wf.read_s8_array(3)
                    normal = Vector(((ny / 127.0), -(nz / 127.0), (nx / 127.0)))
                    if normal.length == 0:
                        normal = Vector((0.0, 0.0, 1.0))
                    else:
                        normal.normalize()
                else:
                    nx, ny, nz = decode_bayo_switch_normal(wf.read_u32())

                    normal = Vector(((nx), (ny), (nz)))
                    if normal.length == 0:
                        normal = Vector((0.0, 0.0, 1.0))
                    else:
                        normal.normalize()'''

                wf.advance(4)  # tangents - skip

                vertices.append((px, py, pz))
                uvs.append((u, 1.0 - v))
                normals.append(normal)

                r, g, b, a = wf.read_u8_array(4)
                extra_vcolors.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))

                if num_uvmaps == 2:
                    u2 = read_half_float(wf.read(2))
                    v2 = read_half_float(wf.read(2))
                    uv2s.append((u2, 1.0 - v2))

                vertex_groups_data.append(list(zip((0,0,0,0), (0,0,0,0))))

        else:
            wf.seek(offset_vertices)
            for i in range(num_vertices):
                px, py, pz = wf.read_f32_vector3()

                u = read_half_float(wf.read(2))
                v = read_half_float(wf.read(2))
                normal = decode_bayonetta_normals(wf, normal_type)
                '''if (normal_type == "B1"):
                    wf.advance(1)
                    nz, ny, nx = wf.read_s8_array(3)
                    normal = Vector(((ny / 127.0), -(nz / 127.0), (nx / 127.0)))
                    if normal.length == 0:
                        normal = Vector((0.0, 0.0, 1.0))
                    else:
                        normal.normalize()
                else:
                    nx, ny, nz = decode_bayo_switch_normal(wf.read_u32())

                    normal = Vector(((nx), (ny), (nz)))
                    if normal.length == 0:
                        normal = Vector((0.0, 0.0, 1.0))
                    else:
                        normal.normalize()'''

                wf.advance(4)  # tangents - skip

                bone_ids = wf.read_u8_array(4)
                bone_weights_raw = wf.read_u8_array(4)

                weights_sum = sum(bone_weights_raw)
                if weights_sum == 0:
                    weights_sum = 1
                bone_weights = [w / weights_sum for w in bone_weights_raw]

                vertices.append((px, py, pz))
                uvs.append((u, 1.0 - v))
                normals.append(normal)
                vertex_groups_data.append(list(zip(bone_ids, bone_weights)))



        if vertexFormat != 0x4000001F and vertexFormat != 0x5800000F and offset_vertices_extra > 0:
            wf.seek(offset_vertices_extra)
            for _ in range(num_vertices):
                r, g, b, a = wf.read_u8_array(4)
                extra_vcolors.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))

                if num_uvmaps == 2:
                    u2 = read_half_float(wf.read(2))
                    v2 = read_half_float(wf.read(2))
                    uv2s.append((u2, 1.0 - v2))


        mesh_batches = []
        current_mesh_pos = offsetMeshes
        wf.seek(offsetMeshesOffsets)
        mesh_offsets = [wf.read_u32() for _ in range(numMeshes)]

        mesh_flags = []
        mesh_datas = []
        #return
        for x in range(numMeshes):
            print(f"[>] Loading Mesh {x}... ", end="")
            wf.seek(offsetMeshes + mesh_offsets[x])
            current_mesh_pos = wf.tell()
            print("Current Pos: " + hex(current_mesh_pos))
            wf.read(2)
            num_batches = wf.read_u16()
            wf.read(4)
            offset_batch_offsets = wf.read_u32()
            mesh_flags.append(wf.read_s32())
            wf.read(16)
            name = wf.read(32).split(b'\x00', 1)[0].decode('ascii')
            mesh_datas.append(struct.unpack(f'{wf.end_flag}3ff3f3fff', wf.read(12 + 4 + 12 + 12 + 8)))
            print(f"{name}")

            batch_offset_table = current_mesh_pos + offset_batch_offsets
            wf.seek(batch_offset_table)
            batch_rel_offsets = [wf.read_u32() for _ in range(num_batches)]
            batch_starts = [batch_offset_table + rel for rel in batch_rel_offsets]


            batch_faces_list = []
            batch_data_list = []
            batch_bone_maps = []

            for batch_start in batch_starts:
                wf.seek(batch_start)
                print(batch_start)
                batch_info = WMBBatchHeader(wf)
                wf.advance(28)
                

                vertex_offset = 0
                if (batch_info.flags & 0x1) != 0:
                    vertex_offset = batch_info.vertexOffset

                num_bone_maps = wf.read_u32()
                bone_map = list(wf.read(num_bone_maps))
                batch_bone_maps.append(bone_map)
                batch_faces = []

                if batch_info.numIndices > 0:
                    wf.seek(batch_start + batch_info.offsetIndices)
                    raw = wf.read(batch_info.numIndices * 2)
                    indices = struct.unpack(f'{wf.end_flag}{batch_info.numIndices}H', raw)

                    if batch_info.primativeType == 4:
                        for i in range(0, len(indices) - 2, 3):
                            batch_faces.append((indices[i] + vertex_offset, indices[i + 2] + vertex_offset, indices[i + 1] + vertex_offset))
                    elif batch_info.primativeType == 5:
                        a, b = indices[0], indices[1]
                        for i in range(2, len(indices)):
                            c = indices[i]
                            if a == b or b == c or c == a:
                                a, b = b, c
                                continue
                            if (i - 2) % 2 == 0:
                                batch_faces.append((b + vertex_offset, a + vertex_offset, c + vertex_offset))
                            else:
                                batch_faces.append((a + vertex_offset, b + vertex_offset, c + vertex_offset))
                            a, b = b, c
                batch_faces_list.append([batch_faces,batch_info])

            mesh_batches.append((name, batch_faces_list, batch_bone_maps, batch_starts, x))
            last_batch_start = batch_starts[-1]
            wf.seek(last_batch_start + 24)
            offset_indices = wf.read_u32()
            num_indices = wf.read_u32()

            index_data_start = last_batch_start + offset_indices
            index_data_end = index_data_start + num_indices * 2

            wf.seek(index_data_end)
            while True:
                marker = wf.read(2)
                if marker != b'\xFB\xFB':
                    wf.seek(-2, 1)
                    break

        for mesh_name, batch_faces_list, batch_bone_maps, batch_starts, mesh_index in mesh_batches:
            for batch_index, batch_faces in enumerate(batch_faces_list, 1):
                if mesh_name.replace("\x00", "") == "":
                    mesh_name=os.path.basename(os.path.splitext(filepath)[0])


                object_name = f"{mesh_index}-{mesh_name}-{batch_index - 1}" # MGR2Blender style
                print(f"[>] Importing {object_name}")
                used_indices = sorted(set(i for tri in batch_faces[0] for i in tri))
                index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
                local_vertices = [vertices[i] for i in used_indices]
                local_uvs = [uvs[i] for i in used_indices]
                local_normals = [normals[i] for i in used_indices]
                local_vertex_groups = [vertex_groups_data[i] for i in used_indices]
                remapped_faces = [
                    (index_remap[a], index_remap[b], index_remap[c]) for a, b, c in batch_faces[0]
                ]

                bone_map = batch_bone_maps[batch_index - 1]

                remapped_vertex_groups = []
                for vg in local_vertex_groups:
                    remapped = []
                    for local_bone_id, weight in vg:
                        if weight > 0 and local_bone_id < len(bone_map):
                            global_bone_id = bone_map[local_bone_id]
                            remapped.append((global_bone_id, weight))
                    remapped_vertex_groups.append(remapped)

                mesh = bpy.data.meshes.new(object_name)

                obj = bpy.data.objects.new(object_name, mesh)

                obj["dummy"] = False
                obj["flags"] = mesh_flags[mesh_index]
                obj["batch_flags"] = batch_faces[1].flags

                obj["unknownE1"] = batch_faces[1].unknownE1
                obj["unknownE2"] = batch_faces[1].unknownE2

                obj["data"] = mesh_datas[mesh_index]
                obj["vertex_start"] = batch_faces[1].vertexStart
                obj["vertex_end"] = batch_faces[1].vertexEnd
                model_collection.objects.link(obj)
                obj.rotation_euler = (math.radians(90), 0, 0)
                if (bayo_2):
                    obj.data.materials.append(bayonettaMaterialList[batch_faces[1].exMaterialID].toBPYMaterial(f"{wmb_name}_{batch_faces[1].exMaterialID}", textures))
                else:
                    obj.data.materials.append(bayonettaMaterialList[batch_faces[1].materialID].toBPYMaterial(f"{wmb_name}_{batch_faces[1].materialID}", textures))

                if (numBones > 0):
                    obj.parent = arm_obj

                    mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                    mod.object = arm_obj


                obj.active_material_index = 0

                if (len(local_vertices) == 1):
                    obj["empty"] = True
                    continue

                mesh.from_pydata(local_vertices, [], remapped_faces)
                mesh.update()

                if (numBones > 0):
                    # vertex groups
                    for vert_index, groups in enumerate(remapped_vertex_groups):
                        for bone_id, weight in groups:
                            group_name = bone_group_name(bone_id)
                            group = obj.vertex_groups.get(group_name)
                            if not group:
                                group = obj.vertex_groups.new(name=group_name)
                            group.add([vert_index], weight, 'REPLACE')

                loop_normals = []
                for face in remapped_faces:
                    for vert_index in face:
                        loop_normals.append(local_normals[vert_index])

                mesh.normals_split_custom_set(loop_normals)

                bm = bmesh.new()
                bm.from_mesh(mesh)

                # UVMap1 (Changed to be in parity with MGR2Blender2MGR)
                if local_uvs:
                    uv_layer = bm.loops.layers.uv.new("UVMap1")
                    for face in bm.faces:
                        for loop in face.loops:
                            loop[uv_layer].uv = local_uvs[loop.vert.index]

                # UVMap2
                local_uv2s = [uv2s[i] for i in used_indices] if uv2s else []
                if local_uv2s:
                    uv_layer_1 = bm.loops.layers.uv.new("UVMap2") 
                    for face in bm.faces:
                        for loop in face.loops:
                            loop[uv_layer_1].uv = local_uv2s[loop.vert.index]


                # primary vertex color
                local_vcolors = [vcolor[i] for i in used_indices] if vcolor else []
                if local_vcolors:
                    color_layer = bm.loops.layers.color.new("Col")
                    for face in bm.faces:
                        for loop in face.loops:
                            loop[color_layer] = local_vcolors[loop.vert.index][:4]

                # extra vertex color
                local_extra_vcolors = [extra_vcolors[i] for i in used_indices] if extra_vcolors else []
                if extra_vcolors:
                    extra_color_layer = bm.loops.layers.color.new("ExCol")
                    for face in bm.faces:
                        for loop in face.loops:
                            loop[extra_color_layer] = local_extra_vcolors[loop.vert.index][:4]

                bm.to_mesh(mesh)
                if "Col" in mesh.vertex_colors:
                    mesh.vertex_colors.active = mesh.vertex_colors["Col"]
                bm.free()

                if (bayo_2):
                    if (batch_faces[1].unknownE1 == 48):
                        obj.hide_set(hide_shadow_meshes)
                else:
                    if (batch_faces[1].unknownE1 == 32 and batch_faces[1].unknownE2 == 15):
                        obj.hide_set(hide_shadow_meshes)


    return {'FINISHED'}

