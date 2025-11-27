import bpy
import struct
import json
from io import BufferedReader
import os
import math
from mathutils import Vector
import bmesh
from .wmb_materials import materialSizeDictionary

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
        self.x = struct.unpack("<f", file.read(4))
        self.y = struct.unpack("<f", file.read(4))
        self.z = struct.unpack("<f", file.read(4))

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

    def __init__(self, file):
        self.batchIdx = struct.unpack("<h", file.read(2))[0]
        self.id = struct.unpack("<h", file.read(2))[0]        
        self.flags = struct.unpack("<H", file.read(2))[0]         
        self.exMaterialID = struct.unpack("<h", file.read(2))[0]   
        self.materialID = struct.unpack("<B", file.read(1))[0]
        self.hasBoneRefs = struct.unpack("<B", file.read(1))[0]
        self.unknownE1 = struct.unpack("<B", file.read(1))[0]
        self.unknownE2 = struct.unpack("<B", file.read(1))[0]
        self.vertexStart = struct.unpack("<I", file.read(4))[0]
        self.vertexEnd = struct.unpack("<I", file.read(4))[0]
        self.primativeType = struct.unpack("<i", file.read(4))[0]
        self.offsetIndices = struct.unpack("<I", file.read(4))[0]
        self.numIndices = struct.unpack("<i", file.read(4))[0]
        self.vertexOffset = struct.unpack("<i", file.read(4))[0]

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

    def __init__(self, file):
        self.id = struct.unpack("<h", file.read(2))[0]
        self.batchCount = struct.unpack("<h", file.read(2))[0]
        self.unknownA1 = struct.unpack("<h", file.read(2))[0]
        self.boundingBoxInfos = struct.unpack("<h", file.read(2))[0]
        self.offsetBatchOffsets = struct.unpack("<I", file.read(4))[0]
        self.flags = struct.unpack("<i", file.read(4))[0]
        file.read(16)
        self.name = file.read(32).decode()
        self.center = WMBVector(file)
        self.height = struct.unpack("<f", file.read(4))[0]
        self.corner1 = WMBVector(file)
        self.corner2 = WMBVector(file)  
        self.unknownD = struct.unpack("<f", file.read(4))[0]
        self.unknownE = struct.unpack("<f", file.read(4))[0]            

class WMBMaterial:
    matID = 0
    flags = 0
    data = 0
    sampler_1_id = 0
    sampler_2_id = 0
    parameter_data = {}

    bpyMaterial = None
    def __init__(self, file, json=None):
        self.matID = struct.unpack("<h", file.read(2))[0]
        self.flags = struct.unpack("<h", file.read(2))[0]
        datasize = materialSizeDictionary[self.matID] - 4
        data_start = file.tell()
        


        if json is not None:
            if str(self.matID) in json:
                mat_info = json[str(self.matID)]
                self.layout = mat_info.get(":layout", {})
                file.seek(data_start)
                for param_name, param_type in self.layout.items():
                    if param_type == "sampler2D_t" or param_type == "samplerCUBE_t":
                        self.parameter_data[param_name] = (struct.unpack("<i", file.read(4)))[0]
                    elif param_type == "f4_float3_t":
                        self.parameter_data[param_name] = (struct.unpack("<fff", file.read(12)))
                        file.read(4)
                    elif param_type == "f4_float2_t":
                        self.parameter_data[param_name] = (struct.unpack("<ff", file.read(8)))
                        file.read(8)
                    elif param_type == "f4_float_t":
                        self.parameter_data[param_name] = (struct.unpack("<f", file.read(4)))[0]
                        file.read(12)
                    else:
                        self.parameter_data[param_name] = (struct.unpack("<ffff", file.read(16)))

                    

            else:
                print("Material type isn't in the JSON... it may be invalid.")
        file.seek(data_start)
        self.data = struct.unpack("<" + str(datasize // 4) + "i", file.read(materialSizeDictionary[self.matID] - 4))
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
            print(f"  {param_name} → {param_type}")


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

        nrm_image_node = nodes.new(type='ShaderNodeTexImage')
        nrm_image_node.location = -100,-500
        if normal_bpy_image is not None:
            nrm_image_node.image = normal_bpy_image
            normalShader = nodes.new(type='ShaderNodeNormalMap')
            normalShader.location = 0, 0
            normalShader.hide = True
            links.new(normalShader.outputs['Normal'], principled.inputs['Normal'])
            invertGreen = invertGreenChannel(nodes, nrm_image_node.location[1])
            links.new(nrm_image_node.outputs["Color"], invertGreen.inputs["Color"])
            links.new(invertGreen.outputs["Color"], normalShader.inputs["Color"])
            links.new(normalShader.outputs["Normal"], principled.inputs["Normal"])




        links.new(alb_image_node.outputs['Color'], principled.inputs["Base Color"])
        wmb_material_list[material_name] = mat




        mat["type"] = self.matID
        mat["flags"] = self.flags
        mat["data"] = self.data
        mat["size"] = materialSizeDictionary[self.matID]
        
        return mat
    
def ImportWMB(filepath, textures=""):
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



    wmb_collection = bpy.data.collections.new("WMB")
    bpy.context.scene.collection.children.link(wmb_collection)
    with open(filepath, 'rb') as f:
        f.seek(8)
        vertexFormat = struct.unpack('<I', f.read(4))[0]
        num_vertices = struct.unpack('<I', f.read(4))[0]
        num_uvmaps = struct.unpack('<B', f.read(1))[0]
        num_colors = struct.unpack('<B', f.read(1))[0]
        f.read(2)
        offset_positions = struct.unpack('<I', f.read(4))[0]
        offset_vertices = struct.unpack('<I', f.read(4))[0]
        offset_vertices_extra = struct.unpack('<I', f.read(4))[0]
        f.read(16)
        numBones = struct.unpack('<I', f.read(4))[0]
        offsetBoneHierarchy = struct.unpack('<I', f.read(4))[0]
        offsetBoneRelativePosition = struct.unpack('<I', f.read(4))[0]
        offsetBonePositions = struct.unpack('<I', f.read(4))[0]
        offsetBoneIndexTranslateTable = struct.unpack('<I', f.read(4))[0]
        numMaterials = struct.unpack('<I', f.read(4))[0]
        offsetMaterialsOffsets = struct.unpack('<I', f.read(4))[0]
        offsetMaterials = struct.unpack('<I', f.read(4))[0]
        numMeshes = struct.unpack('<I', f.read(4))[0]
        offsetMeshesOffsets = struct.unpack('<I', f.read(4))[0]
        offsetMeshes = struct.unpack('<I', f.read(4))[0]
        numPolygons = struct.unpack('<I', f.read(4))[0]
        numShaderSettings = struct.unpack('<I', f.read(4))[0]
        offsetInverseKinematics = struct.unpack('<I', f.read(4))[0]
        offsetBoneSymmetries = struct.unpack('<I', f.read(4))[0]
        offsetBoneFlags = struct.unpack('<I', f.read(4))[0]

        wmb_name = os.path.splitext(os.path.basename(filepath))[0]

        model_collection = bpy.data.collections.new(wmb_name)
        model_collection["vertex_format"] = vertexFormat

        wmb_collection.children.link(model_collection)



        f.seek(offsetMaterialsOffsets)
        bayonettaMaterialList = []
        materialOffsets = struct.unpack(("<" + "I"*numMaterials), f.read(4 * numMaterials))
        for x in range(numMaterials):
            f.seek(offsetMaterials + materialOffsets[x]) # not confusing at all :fire:
            materialData = WMBMaterial(f, material_json)
            bayonettaMaterialList.append(materialData)


        flag_map = {}
        if (offsetBoneFlags != 0):
            f.seek(offsetBoneFlags)
            for i in range(numBones):
                flag_map[i] = struct.unpack("<B", f.read(1))[0]

        # build skel
        if numBones > 0:
            arm_data = bpy.data.armatures.new(wmb_name)
            arm_obj = bpy.data.objects.new(wmb_name, arm_data)
            model_collection.objects.link(arm_obj)
            bpy.context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='EDIT')

            # bone hierarchy
            f.seek(offsetBoneHierarchy)
            bone_parents = [struct.unpack('<h', f.read(2))[0] for _ in range(numBones)]

            arm_obj["bone_hierarchy"] = bone_parents # this really shouldn't be that bad

            # absolute positions
            f.seek(offsetBonePositions)
            bone_abs_positions = [Vector(struct.unpack('<3f', f.read(12))) for _ in range(numBones)]

            # Bullshit bone remap thing, check later
            bone_name_map = {}
            bone_id_map = {}
            if offsetBoneIndexTranslateTable:
                parts_map = []
                f.seek(offsetBoneIndexTranslateTable)

                l1_table = [struct.unpack('<h', f.read(2))[0] for _ in range(16)]

                for l1_index in range(16):
                    l2_offset = l1_table[l1_index]
                    if l2_offset == 0xFFFF:
                        continue

                    for l2_index in range(16):
                        f.seek(offsetBoneIndexTranslateTable + ((l2_offset + l2_index) * 2))
                        l3_offset = struct.unpack('<h', f.read(2))[0]
                        if l3_offset == 0xFFFF:
                            continue

                        for l3_index in range(16):
                            f.seek(offsetBoneIndexTranslateTable + ((l3_offset + l3_index) * 2))
                            parts_index = struct.unpack('<h', f.read(2))[0]
                            if parts_index != 0xFFF:
                                parts_no = (l1_index << 8) | (l2_index << 4) | l3_index
                                parts_map.append((parts_no, parts_index))                            

                for item in parts_map:
                    bone_name_map[item[1]] = f"bone{item[1]:03}"
                    bone_id_map[item[1]] = item[0]


                f.seek(offsetBoneIndexTranslateTable) # TODO, nuke from orbit
                # first level
                first_level = [struct.unpack('<h', f.read(2))[0] for _ in range(16)]
                j = sum(1 for val in first_level if val != -1 and val != 4095)
                arm_obj["translate_table_1"] = first_level
                # second level
                second_level = [struct.unpack('<h', f.read(2))[0] for _ in range(j * 16)]
                k = sum(1 for val in second_level if val != -1 and val != 4095)
                arm_obj["translate_table_2"] = second_level
                # third level
                third_level = [struct.unpack('<h', f.read(2))[0] for _ in range(k * 16)]
                arm_obj["translate_table_3"] = third_level
                arm_obj["translate_table_size"] = f.tell() - offsetBoneIndexTranslateTable

                        
            print(bone_name_map)
            edit_bones = {}
            for i in range(numBones):
                bone_name = bone_name_map.get(i, f"bone{i:03}")
                bone_id = bone_id_map.get(i, -1)
                bone = arm_data.edit_bones.new(bone_name)
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
            f.seek(offsetInverseKinematics)
            kincount = struct.unpack("<b", f.read(1))[0]
            other_data = struct.unpack("<bbb", f.read(3))
            offset = struct.unpack("<i", f.read(4))[0]
            arm_obj["ik_count"] = kincount
            arm_obj["ik_offset"] = offset
            arm_obj["ik_unk"] = other_data
            f.seek(offsetInverseKinematics + offset)
            for i in range(kincount):
                structure = struct.unpack("<" + ("b" * 16), f.read(16))
                arm_obj["ik_structure_" + str(i)] = structure



        # Read vertex positions separately if offset_positions is non-zero
        vertices, vcolor, uvs, normals, vertex_groups_data = [], [], [], [], []

        if vertexFormat == 0x4000001F:
            # read positions from offset_positions
            f.seek(offset_positions)
            positions = [struct.unpack('<3f', f.read(12)) for _ in range(num_vertices)]

            # now read the rest of the vertex data
            f.seek(offset_vertices)

            for i in range(num_vertices):
                px, py, pz = positions[i]

                # Normal as 3 floats
                nx, ny, nz = struct.unpack('<3f', f.read(12))
                normal = Vector((nx, ny, nz))
                if normal.length == 0:
                    normal = Vector((0.0, 0.0, 1.0))
                else:
                    normal.normalize()

                r, g, b, a = struct.unpack('<4B', f.read(4))
                vcolor.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))
                f.read(4)  # tangents - skip

                bone_ids = struct.unpack('<4B', f.read(4))
                bone_weights_raw = struct.unpack('<4B', f.read(4))

                u, v = struct.unpack('<2f', f.read(8))  # UVs as full floats

                weights_sum = sum(bone_weights_raw)
                if weights_sum == 0:
                    weights_sum = 1
                bone_weights = [w / weights_sum for w in bone_weights_raw]

                vertices.append((px, py, pz))
                uvs.append((u, 1.0 - v))
                normals.append(normal)
                vertex_groups_data.append(list(zip(bone_ids, bone_weights)))

        else:
            f.seek(offset_vertices)

            for i in range(num_vertices):
                px, py, pz = struct.unpack('<3f', f.read(12))

                u = read_half_float(f.read(2))
                v = read_half_float(f.read(2))
                f.read(1)
                nz, ny, nx = struct.unpack('<3b', f.read(3))
                normal = Vector((nx / 127.0, ny / 127.0, nz / 127.0))
                if normal.length == 0:
                    normal = Vector((0.0, 0.0, 1.0))
                else:
                    normal.normalize()

                f.read(4)  # tangents - skip

                bone_ids = struct.unpack('<4B', f.read(4))
                bone_weights_raw = struct.unpack('<4B', f.read(4))

                weights_sum = sum(bone_weights_raw)
                if weights_sum == 0:
                    weights_sum = 1
                bone_weights = [w / weights_sum for w in bone_weights_raw]

                vertices.append((px, py, pz))
                uvs.append((u, 1.0 - v))
                normals.append(normal)
                vertex_groups_data.append(list(zip(bone_ids, bone_weights)))

        extra_vcolors = []
        uv2s = []

        if vertexFormat != 0x4000001F and offset_vertices_extra > 0:
            f.seek(offset_vertices_extra)
            for _ in range(num_vertices):
                r, g, b, a = struct.unpack('<4B', f.read(4))
                extra_vcolors.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))

                if num_uvmaps == 2:
                    u2 = read_half_float(f.read(2))
                    v2 = read_half_float(f.read(2))
                    uv2s.append((u2, 1.0 - v2))


        mesh_batches = []
        current_mesh_pos = offsetMeshes
        f.seek(offsetMeshesOffsets)
        mesh_offsets = [struct.unpack('<I', f.read(4))[0] for _ in range(numMeshes)]

        mesh_flags = []
        mesh_datas = []
        #return
        for x in range(numMeshes):
            f.seek(offsetMeshes + mesh_offsets[x])
            current_mesh_pos = f.tell()
            
            f.read(2)
            num_batches = struct.unpack('<H', f.read(2))[0]
            f.read(4)
            offset_batch_offsets = struct.unpack('<I', f.read(4))[0]
            mesh_flags.append(struct.unpack('<i', f.read(4))[0])
            f.read(16)
            name = f.read(32).split(b'\x00', 1)[0].decode('ascii')
            mesh_datas.append(struct.unpack('<3ff3f3fff', f.read(12 + 4 + 12 + 12 + 8)))
            print(f"[>] Loading Mesh: {name}")
            while True:
                val = f.read(2)
                if val != b'\xFB\xFB':
                    f.seek(-2, 1)
                    break
            batch_offset_table = current_mesh_pos + offset_batch_offsets
            f.seek(batch_offset_table)
            batch_rel_offsets = [struct.unpack('<I', f.read(4))[0] for _ in range(num_batches)]
            batch_starts = [batch_offset_table + rel for rel in batch_rel_offsets]


            batch_faces_list = []
            batch_data_list = []
            batch_bone_maps = []

            for batch_start in batch_starts:
                f.seek(batch_start)
                batch_info = WMBBatchHeader(f)
                f.read(28)

                num_bone_maps = struct.unpack('<I', f.read(4))[0]
                bone_map = list(f.read(num_bone_maps))
                batch_bone_maps.append(bone_map)

                batch_faces = []

                if batch_info.numIndices > 0:
                    f.seek(batch_start + batch_info.offsetIndices)
                    raw = f.read(batch_info.numIndices * 2)
                    indices = struct.unpack(f'<{batch_info.numIndices}H', raw)

                    if batch_info.primativeType == 4:
                        for i in range(0, len(indices) - 2, 3):
                            batch_faces.append((indices[i], indices[i + 1], indices[i + 2]))
                    elif batch_info.primativeType == 5:
                        a, b = indices[0], indices[1]
                        for i in range(2, len(indices)):
                            c = indices[i]
                            if a == b or b == c or c == a:
                                a, b = b, c
                                continue
                            if (i - 2) % 2 == 0:
                                batch_faces.append((b, a, c))
                            else:
                                batch_faces.append((a, b, c))
                            a, b = b, c
                batch_faces_list.append([batch_faces,batch_info])

            mesh_batches.append((name, batch_faces_list, batch_bone_maps, batch_starts, x))
            print(batch_starts)
            last_batch_start = batch_starts[-1]
            f.seek(last_batch_start + 24)
            offset_indices = struct.unpack('<I', f.read(4))[0]
            num_indices = struct.unpack('<I', f.read(4))[0]

            index_data_start = last_batch_start + offset_indices
            index_data_end = index_data_start + num_indices * 2

            f.seek(index_data_end)
            while True:
                marker = f.read(2)
                if marker != b'\xFB\xFB':
                    f.seek(-2, 1)
                    break

        for mesh_name, batch_faces_list, batch_bone_maps, batch_starts, mesh_index in mesh_batches:
            for batch_index, batch_faces in enumerate(batch_faces_list, 1):
                object_name = f"{mesh_index}-{mesh_name}-{batch_index - 1}" # MGR2Blender style
                print(f"[>] Importing {object_name}")
                print(batch_faces)
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
                obj["flags"] = mesh_flags[mesh_index]
                obj["batch_flags"] = batch_faces[1].flags
                obj["data"] = mesh_datas[mesh_index]
                obj["vertex_start"] = batch_faces[1].vertexStart
                obj["vertex_end"] = batch_faces[1].vertexEnd
                model_collection.objects.link(obj)
                obj.rotation_euler = (math.radians(90), 0, 0)
                obj.data.materials.append(bayonettaMaterialList[batch_faces[1].materialID].toBPYMaterial(f"{wmb_name}_{batch_faces[1].materialID}", textures))
                obj.parent = arm_obj

                mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                mod.object = arm_obj
                obj.active_material_index = 0

                if (len(local_vertices) == 1):
                    obj["empty"] = True
                    continue

                mesh.from_pydata(local_vertices, [], remapped_faces)
                mesh.update()

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
                    extra_color_layer = bm.loops.layers.color.new("Col")
                    for face in bm.faces:
                        for loop in face.loops:
                            loop[extra_color_layer] = local_extra_vcolors[loop.vert.index][:4]

                bm.to_mesh(mesh)
                if "Col" in mesh.vertex_colors:
                    mesh.vertex_colors.active = mesh.vertex_colors["Col"]
                bm.free()
    return {'FINISHED'}

