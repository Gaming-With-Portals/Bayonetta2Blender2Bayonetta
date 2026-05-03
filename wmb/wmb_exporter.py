import bpy
import struct
from collections import defaultdict
import bmesh
from io import BufferedReader
from mathutils import Vector, Matrix
import numpy as np
from .wmb_materials import materialSizeDictionary
from .wmb_custom_bones import encode_parts_index_no_table as GenerateTranslateTable
import re

GENERATE_TRANSLATE_TABLE = True
USE_LARGE_BONES = False
OP_INSTANCE = None
EXCEPT_AFTER_GENERATION = False
COPY_UV_1_AS_2 = True
BAYONETTA_2 = False
EXPORT_AS_STATIC_MESH = False

bone_name_to_id_map = {}

def align(offset, alignment):
    return offset if offset % alignment == 0 else offset + (alignment - (offset % alignment))

def float_to_half_bytes(f):
    h = np.float16(f)
    return h.tobytes()

def reportError(err_string):
    global EXCEPT_AFTER_GENERATION
    global OP_INSTANCE

    print(f"ERROR: {err_string}")
    OP_INSTANCE.report({'ERROR'}, err_string)
    EXCEPT_AFTER_GENERATION = True

def pack_b2_normal(fx, fy, fz):
    mag = (1<<9)-1
    def pack_component(f):
        i = int(f * float(mag))
        i = min(max(i, -mag), mag)
        return i & ((1 << 10) - 1) 

    nx = pack_component(fx)
    ny = pack_component(fy)
    nz = pack_component(fz)
    mask = (1<<10)-1
    v = 0
    v |= nz & mask
    v <<= 10
    v |= ny & mask
    v <<= 10
    v |= nx & mask


    return v

def getObjectChildren(parent):
    if (EXPORT_AS_STATIC_MESH):
        return parent.objects
    else:
        return parent.children

def decimalFixup(name):
    return re.sub(r'\.\d+$', '', name)

def get_blenderUVCoordsEx(self, objOwner, loopIndex, uvName): # From MGR2Blender, but I wrote it, so is it really?
    uv_layers = objOwner.data.uv_layers

    if uvName not in uv_layers:
        return None

    uv_coords = uv_layers[uvName].data[loopIndex].uv
    return (uv_coords.x, 1 - uv_coords.y)

def blenderColorToBayo(color):
    r = int(color[0]*255)
    g = int(color[1]*255)
    b = int(color[2]*255)
    a = int(color[3]*255)
    return (r, g, b, a)

class WMBVertexChunk:


    def __init__(self, children, ref_table, b2):
        self.vertex_infos = []
        self.exvertex_infos = []
        self.total_vertices = 0
        self.num_mapping = 2
        self.num_color = 1

        if (b2):
            self.num_mapping = 1

        self.exvertex_size = (self.num_color*4)
        if (self.num_mapping>1):
            self.exvertex_size+=(self.num_mapping-1)*4
        
        vertex_ticker = 0
        for obj in children:
            if obj.type != 'MESH':
                continue
            
            print(f"[>] Generating vertex data for {obj.name}")

            if len(obj.data.uv_layers) != 0:
                obj.data.calc_tangents()

            def get_blenderLoops(self, objOwner):
                blenderLoops = []
                blenderLoops += objOwner.data.loops

                return blenderLoops

            ref_table[obj.name] = {}
            bone_counter = 0

            loops = get_blenderLoops(self, obj)
            sorted_loops = sorted(loops, key=lambda loop: loop.vertex_index)

            ex_color_layer = obj.data.vertex_colors.get("ExCol", None)
            main_color_layer = obj.data.vertex_colors.get("Col", None)

            obj["vertex_start"] = vertex_ticker

            previousIndex = -1
            for loop in sorted_loops:
                vertex_info = []
                ex_vertex_info = []
                if (loop.vertex_index == previousIndex):
                    continue

                vertex_ticker+=1

                previousIndex = loop.vertex_index

                bvtx = obj.data.vertices[loop.vertex_index]

                MAX_WEIGHTS = 4
                bone_weights = []
                bone_indices = []
                for g in bvtx.groups:
                    group_index = g.group
                    weight = g.weight
                    group_name = obj.vertex_groups[group_index].name
                    bone_id = 0

                    if (getBoneID(group_name) in ref_table[obj.name]):
                        bone_id = ref_table[obj.name][getBoneID(group_name)]
                    else:
                        bone_id = bone_counter
                        ref_table[obj.name][getBoneID(group_name)] = bone_counter
                        bone_counter+=1

                    
                    bone_weights.append(weight)
                    bone_indices.append(bone_id)

                if (not EXPORT_AS_STATIC_MESH):
                    pairs = list(zip(bone_weights, bone_indices))
                    pairs = sorted(pairs, key=lambda p: p[0], reverse=True)[:MAX_WEIGHTS]

                    float_weights = [p[0] for p in pairs]
                    sel_indices = [p[1] for p in pairs]

                    total = sum(float_weights)
                    if total > 0:
                        float_weights = [w / total for w in float_weights]
                    else:
                        float_weights = [0.0] * len(float_weights)

                    while len(float_weights) < MAX_WEIGHTS:
                        float_weights.append(0.0)
                        sel_indices.append(0)

                    int_weights = [int(w * 255.0) for w in float_weights]

                    rem = 255 - sum(int_weights)
                    if rem != 0:
                        order = sorted(range(MAX_WEIGHTS), key=lambda i: float_weights[i], reverse=True)
                        i = 0
                        while rem != 0:
                            idx = order[i]
                            if rem > 0 and int_weights[idx] < 255:
                                int_weights[idx] += 1
                                rem -= 1
                            elif rem < 0 and int_weights[idx] > 0:
                                int_weights[idx] -= 1
                                rem += 1
                            i = (i + 1) % MAX_WEIGHTS

                    int_weights = [max(0, min(255, w)) for w in int_weights]


                position = (bvtx.co.x, bvtx.co.y, bvtx.co.z)
                vertex_info.append(position) # Write position to buffer

                normal = (loop.normal[0], loop.normal[2], -loop.normal[1])

                
                vertex_info.append(normal) # Write normal to buffer
                


                loopTangent = loop.tangent * 127
                tx = int(loopTangent[0] + 127.0)
                ty = int(loopTangent[1] + 127.0)
                tz = int(loopTangent[2] + 127.0)
                sign = 0xff if loop.bitangent_sign == -1 else 0
                vertex_info.append((tx, ty, tz, sign))

                if (not EXPORT_AS_STATIC_MESH): # This is so bad LMAO
                    vertex_info.append(tuple(sel_indices))
                    vertex_info.append(tuple(int_weights))
                else:
                    vertex_info.append((0, 0, 0, 0))
                    vertex_info.append((0, 0, 0, 0))

                mainUV = get_blenderUVCoordsEx(self, obj, loop.index, "UVMap1")
                vertex_info.append(mainUV)
                vertex_info.append([]) # Unused
                if (main_color_layer is not None):
                    vertex_info.append(blenderColorToBayo((main_color_layer.data[loop.index].color)))


                exMap = get_blenderUVCoordsEx(self, obj, loop.index, "UVMap2")


                if (ex_color_layer is not None):
                    ex_vertex_info.append(blenderColorToBayo((ex_color_layer.data[loop.index].color)))
                else:
                    ex_vertex_info.append((0, 0, 0, 0))

                if (self.num_mapping == 2):
                    if (COPY_UV_1_AS_2):
                        ex_vertex_info.append(mainUV)
                    else:
                        if (exMap is not None):
                            ex_vertex_info.append(exMap)
                    
                    

                    


                self.vertex_infos.append(vertex_info)
                self.exvertex_infos.append(ex_vertex_info)

                if (obj["dummy"]):
                    break # Quit after the first iteration

            obj["vertex_end"] = vertex_ticker
        self.total_vertices = vertex_ticker



        '''ref_table[obj.name] = {}
            bone_counter = 0

            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            
            eval_mesh = eval_obj.to_mesh()

            if (eval_mesh.uv_layers.active is not None):
                eval_mesh.calc_tangents(uvmap=eval_mesh.uv_layers.active.name)

            
            sorted_loops = sorted(eval_mesh.loops, key=lambda loop: loop.vertex_index)

            color_layer = eval_mesh.vertex_colors.get("ExCol", None)
            main_color_layer = eval_mesh.vertex_colors.get("Col", None)
            
            uv_layer = eval_mesh.uv_layers.active
            if (uv_layer is None):
                print("Warning! Mesh will not have UVs as there is no active UV layer")

            uv_layer_2 = eval_mesh.uv_layers.get("UVMap2", None)
            loop_map = {} 
            loop_map_2 = {}
            tangent_map = {}
            color_map = {}
            main_color_map = {}

            for loop in eval_mesh.loops:
                vidx = loop.vertex_index
                if vidx not in loop_map and uv_layer is not None:
                    loop_map[vidx] = uv_layer.data[loop.index].uv.copy() 
                
                if vidx not in loop_map_2 and uv_layer_2 is not None:
                    loop_map_2[vidx] = uv_layer_2.data[loop.index].uv.copy() 

                if vidx not in color_map and color_layer is not None:
                    color_map[vidx] = color_layer.data[loop.index].color[:]

                if vidx not in main_color_map and main_color_layer is not None:
                    main_color_map[vidx] = main_color_layer.data[loop.index].color[:]

                if vidx not in tangent_map:
                    tangent = loop.tangent
                    normal = loop.normal


                    sign = 0xff if loop.bitangent_sign == -1 else 0
                    tangent_map[vidx] = (tangent.x, tangent.y, tangent.z, sign)



            obj["vertex_start"] = vertex_ticker

            self.total_vertices += len(eval_mesh.vertices)

            for vertex in eval_mesh.vertices:
                vertex_ticker+=1
                pos = vertex.co
                normal = vertex.normal  # vertex normal (not loop normal)
                
                uv = loop_map.get(vertex.index, Vector((0.0, 0.0)))
                uv_2 = loop_map_2.get(vertex.index, Vector((0.0, 0.0)))
                MAX_WEIGHTS = 4
                bone_weights = []
                bone_indices = []
                for g in vertex.groups:
                    group_index = g.group
                    weight = g.weight
                    group_name = obj.vertex_groups[group_index].name
                    bone_id = 0

                    if (getBoneID(group_name) in ref_table[obj.name]):
                        bone_id = ref_table[obj.name][getBoneID(group_name)]
                    else:
                        bone_id = bone_counter
                        ref_table[obj.name][getBoneID(group_name)] = bone_counter
                        bone_counter+=1

                    
                    bone_weights.append(weight)
                    bone_indices.append(bone_id)

                if (not EXPORT_AS_STATIC_MESH):
                    pairs = list(zip(bone_weights, bone_indices))
                    pairs = sorted(pairs, key=lambda p: p[0], reverse=True)[:MAX_WEIGHTS]

                    float_weights = [p[0] for p in pairs]
                    sel_indices = [p[1] for p in pairs]

                    total = sum(float_weights)
                    if total > 0:
                        float_weights = [w / total for w in float_weights]
                    else:
                        float_weights = [0.0] * len(float_weights)

                    while len(float_weights) < MAX_WEIGHTS:
                        float_weights.append(0.0)
                        sel_indices.append(0)

                    int_weights = [int(w * 255.0) for w in float_weights]

                    rem = 255 - sum(int_weights)
                    if rem != 0:
                        order = sorted(range(MAX_WEIGHTS), key=lambda i: float_weights[i], reverse=True)
                        i = 0
                        while rem != 0:
                            idx = order[i]
                            if rem > 0 and int_weights[idx] < 255:
                                int_weights[idx] += 1
                                rem -= 1
                            elif rem < 0 and int_weights[idx] > 0:
                                int_weights[idx] -= 1
                                rem += 1
                            i = (i + 1) % MAX_WEIGHTS

                    int_weights = [max(0, min(255, w)) for w in int_weights]


                def blenderColorToBayo(color):
                    r = int(color[0]*255)
                    g = int(color[1]*255)
                    b = int(color[2]*255)
                    a = int(color[3]*255)
                    return (r, g, b, a)

                vertex_info = []
                ex_vertex_info = []
                vertex_info.append(pos.copy())
                vertex_info.append((normal.x, normal.z, -normal.y))

                tangent_info = tangent_map.get(vertex.index, (0.0, 0.0, 0.0, 1.0))

                vertex_info.append(tangent_info)
                if (not EXPORT_AS_STATIC_MESH): # This is so bad LMAO
                    vertex_info.append(tuple(sel_indices))
                    vertex_info.append(tuple(int_weights))
                else:
                    vertex_info.append((0, 0, 0, 0))
                    vertex_info.append((0, 0, 0, 0))

                vertex_info.append(uv.copy())
                vertex_info.append((uv[0], uv[1]))
                if (main_color_layer):
                    vertex_info.append(blenderColorToBayo(main_color_layer.get(vertex.index, (0, 0, 0, 0))))

                ex_vertex_info.append(blenderColorToBayo(color_map.get(vertex.index, (0, 0, 0, 0))))
                if (self.num_mapping == 2):
                    if (COPY_UV_1_AS_2):
                        ex_vertex_info.append(uv.copy())
                    else:
                        ex_vertex_info.append(uv_2.copy())

                self.vertex_infos.append(vertex_info)
                self.exvertex_infos.append(ex_vertex_info)

                if (obj["dummy"]):
                    break # Quit after the first iteration

            obj["vertex_end"] = vertex_ticker'''

def getBoneID(boneName):
    return bone_name_to_id_map[boneName] # later is now, and this is important!

    #return int(boneName[4:]) # This will be important later



class WMBWeightDataVertexChunk:
        def __init__(self, children, vertexchunk : WMBVertexChunk, meshes):

            self.vertex_infos = vertexchunk.vertex_infos
            for obj in children:
                if obj.type != 'MESH':
                    continue
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = obj.evaluated_get(depsgraph)
                eval_mesh = eval_obj.to_mesh()

                current_batch = obj.name.split("-")[0]

                for vertex in eval_mesh.vertices:
                    MAX_WEIGHTS = 4
                    bone_weights = []
                    bone_indices = []
                    for g in vertex.groups:
                        group_index = g.group
                        weight = g.weight
                        group_name = obj.vertex_groups[group_index].name
                        bone_id = getBoneID(group_name)
                        bone_weights.append(weight)
                        bone_indices.append(bone_id)

                    bone_data = sorted(zip(bone_weights, bone_indices), reverse=True)[:MAX_WEIGHTS]
                    weights, indices = zip(*bone_data) if bone_data else ([], [])
                    weights = list(weights) + [0.0] * (MAX_WEIGHTS - len(weights))
                    
                    indices = list(indices) + [0] * (MAX_WEIGHTS - len(indices))



class WMBBoneParents:
    def __init__(self, arm_obj):
        self.bone_map = {}
        for bone in arm_obj.data.bones:
            if bone.parent:
                self.bone_map[getBoneID(bone.name)] = int(getBoneID(bone.parent.name))
            else:
                self.bone_map[getBoneID(bone.name)] = -1

class WMBBonePosition:
    def __init__(self, arm_obj):
        print("[>] Generating bone position data...")
        self.bone_rel_map = {}
        self.bone_pos_map = {}

        for bone in arm_obj.data.bones:
            parent_pos = Vector((0, 0, 0))
            if bone.parent:
                parent_pos = bone.parent.head_local
            
            bone_rel_pos = bone.head_local - parent_pos
            bone_abs_pos = bone.head_local

            self.bone_rel_map[getBoneID(bone.name)] = (bone_rel_pos.x, bone_rel_pos.y, bone_rel_pos.z)
            self.bone_pos_map[getBoneID(bone.name)] = (bone_abs_pos.x, bone_abs_pos.y, bone_abs_pos.z)



class WMBBoneIndexTranslateTable:
    def __init__(self, arm_obj):
        if (GENERATE_TRANSLATE_TABLE):
            translate_table_food = []

            for bone in sorted(arm_obj.data.bones, key=lambda x: x["id"]):
                translate_table_food.append((bone["id"], getBoneID(bone.name)))


            self.data = GenerateTranslateTable(translate_table_food)
            self.size = len(self.data) * 2

        else:
            self.level_1 = arm_obj["translate_table_1"]
            self.level_2 = arm_obj["translate_table_2"]
            self.level_3 = arm_obj["translate_table_3"]
            self.size = (len(arm_obj["translate_table_1"]) * 2) + (len(arm_obj["translate_table_2"]) * 2) + (len(arm_obj["translate_table_3"]) * 2)

class WMBInverseKinetic:
    def __init__(self, arm_obj):
        self.enabled = arm_obj["inverse_kinematics"]
        self.size = 0

        if (self.enabled):
            self.size+=8 # Header
            self.count = arm_obj["ik_count"]
            self.data = arm_obj["ik_unk"]
            self.offset = arm_obj["ik_offset"]
            self.structures = []
            for i in range(self.count):
                self.size+=16
                self.structures.append(arm_obj["ik_structure_" + str(i)])

class WMBBoneSymmetries:
    def __init__(self, arm_obj):
        # TODO: Not perfect
        self.enabled = False
        self.sym_map = {}
        if (arm_obj["bone_symmetries"]):
            self.enabled = True
            
            for bone in arm_obj.data.bones: # this is such a terrible way of doing this lmao
                pos = bone.head_local
                for bone_2 in arm_obj.data.bones:
                    if bone.name == bone_2.name:
                        continue

                    pos_2 = bone_2.head_local
                    if (pos[0] == 0 or pos_2[0] == 0): # Skip center bones
                        continue


                    if abs(-pos[0] - pos_2[0]) < 0.0001:
                        self.sym_map[getBoneID(bone.name)] = getBoneID(bone_2.name)

class WMBBoneFlags:
    def __init__(self, arm_obj):
        self.enabled = False
        self.flag_map = {}
        if (arm_obj["bone_flags"]):
            bpy.context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='EDIT')
            self.enabled = True
            for bone in arm_obj.data.edit_bones:
                if "flags" in bone:
                    self.flag_map[getBoneID(bone.name)] = bone["flags"]
                else:
                    self.flag_map[getBoneID(bone.name)] = 5

            bpy.ops.object.mode_set(mode='OBJECT')

class WMBMaterial(): # Good enough for a direct port 
    def __init__(self):
        self.id = 0
        self.flag = 0
        self.size = 0
        self.type = 0
        self.formal_data = []
        self.data = []

    def fetch_size(self):
        return self.size
    
    def write(self, f):
        f.write(struct.pack("<H", self.type))
        f.write(struct.pack("<H", self.flag))
        for data in self.formal_data:
            data_fmt = data.type
            if (data_fmt == "sampler2D_t" or data_fmt == "samplerCUBE_t"):
                f.write(struct.pack("<i", data.value_int))
            elif (data_fmt == "f4_float3_t"):
                f.write(struct.pack("<fff", *data.value_vec3))
                f.write(struct.pack("<f", -1))
            elif (data_fmt == "f4_float2_t"):
                f.write(struct.pack("<ff", *data.value_vec2))
                f.write(struct.pack("<f", -1))
                f.write(struct.pack("<f", -1))
            elif (data_fmt == "f4_float_t"):
                f.write(struct.pack("<f", data.value_float))
                f.write(struct.pack("<f", -1))
                f.write(struct.pack("<f", -1))
                f.write(struct.pack("<f", -1))
            else:
                f.write(struct.pack("<ffff", *data.value_vec4))

        #for i in self.data: # Write raw data
        #    f.write(struct.pack("<i", i))

class Bayonetta2Material():
    def __init__(self):
        self.id = 0
        self.flag = 0
        self.datas = []
        self.exdatas = []
        self.shader_name = []

    def fetch_size(self):
        return 0x4 + (0x4*len(self.datas))+(0x4*len(self.exdatas))
    
    def write(self, f):
        f.write(struct.pack("<H", self.id))
        f.write(struct.pack("<H", self.flag))
        for data in self.datas:
            f.write(struct.pack("<I", data))
        for data in self.exdatas:
            f.write(struct.pack("<f", data))

class WMBMaterialBlob:
    def __init__(self, arm_obj, material_map, bayonetta_2):

        self.materials = []

        if (not bayonetta_2):
            for i, mat in enumerate(material_map):
                emat = WMBMaterial()
                emat.id = i
                emat.size = materialSizeDictionary[mat.bayo_data.type]
                emat.flag = mat.bayo_data.flags
                emat.type = mat.bayo_data.type
                #emat.data = mat["data"]
                emat.formal_data = mat.bayo_data.parameters
                self.materials.append(emat)

        else:
            for i, mat in enumerate(material_map):
                emat = Bayonetta2Material()

                texture_storage_info = {}
                for tex in mat.bayo_data.textures:
                    texture_storage_info[tex.position] = int(tex.data)
                for dat in mat.bayo_data.b2_data:
                    texture_storage_info[dat.position] = struct.unpack("<I", struct.pack("<f", dat.data))[0]

                for x in range(len(texture_storage_info.keys())):
                    emat.datas.append(texture_storage_info[x])

                for exdat in mat.bayo_data.ex_material_data:
                    emat.exdatas.append(exdat.data)
                

                emat.id = i
                emat.flag = mat.get("flags", 0)
                '''emat.textures = mat.get("raw_data", [])
                emat.datas = mat.get("data", [])'''
                emat.shader_name = mat.bayo_data.shader
                self.materials.append(emat)


        self.material_count = len(self.materials)
        self.offsets = []
        offset_ticker = 0
        for mat in self.materials:
            self.offsets.append(offset_ticker)
            offset_ticker+=mat.fetch_size()

        self.total_size = offset_ticker


class WMBMeshBlob():
    def __init__(self, arm_obj):
        self.mesh_count = 0
        self.offsets = []
        self.meshes = []

        
        for obj in getObjectChildren(arm_obj):
            if obj.type != 'MESH':
                continue
            name_parts = obj.name.split("-")
            if (len(name_parts) == 3):
                if (int(decimalFixup(name_parts[2])) == 0):
                    self.mesh_count+=1

class WMBBatch():
    def __init__(self, parent, obj, bone_ref_table, material_remap, b2):
        if ("dummy" in obj):
            self.is_dummy = obj["dummy"]
        else:
            self.is_dummy = False
       
        if ("batch_flags" not in obj):
            obj["batch_flags"] = 32769
        if ("material_id" not in obj):
            obj["material_id"] = 0
        if ("unknownE1" not in obj):
            obj["unknownE1"] = 0
        if ("unknownE2" not in obj):
            obj["unknownE2"] = 0


        self.batch_idx = 0
        self.id = parent.mesh_id
        self.flags = obj.get("batch_flags", 32769)
        self.exmaterial_id = 0
        self.material_id = 0
        try:
            material_index = material_remap.index(obj.material_slots[0].material)
        except:
            print(f"[!] Error, {obj.name}'s material somehow wasn't remapped??")
            material_index = 0

        if (b2):
            self.exmaterial_id = int(material_index)
        else:
            self.material_id = int(material_index)

        self.has_bone_refs = 0
        self.vertex_start = obj["vertex_start"]
        if (self.is_dummy):
            self.vertex_end = obj["vertex_start"]+1
        else:
            self.vertex_end = obj["vertex_end"]

        self.vertex_offset = self.vertex_start
        self.flags |= 1
        
        self.primitive_type = 4
        
        batch_ref_table = bone_ref_table[obj.name]

        self.unknownE1 = obj.get("unknownE1", 0)
        self.unknownE2 = obj.get("unknownE2", 0)
        

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)  # Convert to tris
        bm.to_mesh(mesh)
        bm.free()
        mesh.calc_loop_triangles()
        self.indices = []
        '''for tri in mesh.loop_triangles:
            self.indices.extend([
                tri.vertices[0],
                tri.vertices[1],
                tri.vertices[2],
            ])'''
        
        if (self.is_dummy):
            self.indices.extend([
                0,
                0,
                0,
            ])
        else:
            for tri in mesh.loop_triangles:
                self.indices.extend([
                    tri.vertices[0],
                    tri.vertices[2],
                    tri.vertices[1],
                ])
        


        if (EXPORT_AS_STATIC_MESH):
            self.has_bone_refs = 0
            self.required_bones = []
        else:
            self.has_bone_refs = 1
            self.required_bones = []
            
            for global_id, local_id in sorted(batch_ref_table.items(), key=lambda x: x[1]):
                self.required_bones.append(global_id)

            if (self.is_dummy):
                self.required_bones = [0]


        '''for group in obj.vertex_groups:
            self.has_bone_refs = 1
            #self.required_bones.append(int(group.name[4:]))
            bone_id = int(group.name[4:])
            while len(self.required_bones) <= bone_id:
                self.required_bones.append(0)  # pad with 0s

            self.required_bones[bone_id] = bone_id'''

        if (USE_LARGE_BONES):
            self.indice_offset = align(0x40 + 0x8 + len(self.required_bones) * 2, 0x80)
        else:
            self.indice_offset = align(0x40 + 0x4 + len(self.required_bones), 0x80)

        print(f"[>] Generated batch from {obj.name} with material {self.material_id}/{self.exmaterial_id}")


    def fetch_size(self):
        size = self.indice_offset
        size+=len(self.indices)*2

        return size
        


class WMBMesh():
    def __init__(self, arm_obj, obj, bone_ref_table, material_remap, b2):
        if ("data" not in obj):
            obj["data"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if ("flags" not in obj):
            obj["flags"] = -2147483648


        name_parts = obj.name.split("-")
        self.name = name_parts[1]
        self.exdata = obj["data"]
        self.mesh_id = int(name_parts[0])
        bpy_batches = []
        self.batches = []


        self.center = (obj["data"][0], obj["data"][1], obj["data"][2])
        self.height = obj["data"][3]
        self.corner1 = (obj["data"][4], obj["data"][5], obj["data"][6])
        self.corner2 = (obj["data"][7], obj["data"][8], obj["data"][9])

        for obj in getObjectChildren(arm_obj):
            if obj.type != 'MESH':
                continue
            name_parts = obj.name.split("-")
            if (int(name_parts[0]) == self.mesh_id):
                bpy_batches.append((int(decimalFixup(name_parts[2])), obj))

        bpy_batches.sort(key=lambda x: x[0])
        bpy_batches = [obj for _, obj in bpy_batches]

        self.batch_count = len(bpy_batches)
        self.flags = -2147483648
        self.batch_offset_offset = 128

        if (b2):
            self.bounding_box_infos = 2
        else:
            self.bounding_box_infos = 1

        self.batch_start_offset = 4 * self.batch_count
        offset = self.batch_start_offset
        self.batch_offsets = []
        for batch_obj in bpy_batches:
            offset = align(offset, 32)
            tmp_batch = WMBBatch(self, batch_obj, bone_ref_table, material_remap, b2)
            self.batches.append(tmp_batch)
            self.batch_offsets.append(offset)
            offset+=tmp_batch.fetch_size()

        print(f"[>] Generated mesh from {self.name} with {len(self.batches)} batch(es)")

    def fetch_size(self):
        size = self.batch_offset_offset
        size += self.batch_start_offset
        for batch in self.batches:
            size = align(size, 32)
            size += batch.fetch_size()
        return size

# -- Bayonetta 2 --
class WMBExMaterialInfo():
    def __init__(self, arm_obj, materials):
        self.tex_info = set()

        for mat in materials:
            for tex in mat.bayo_data.textures:
                self.tex_info.add((int(tex.data), int(tex.flag)))
        

class WMBDataGenerator:
    def __init__(self, colName="WMB", targetCollection=None):
        ALIGN_TARGET = 64


        if (targetCollection == None):
            wmb_collection =  bpy.context.view_layer.layer_collection.children[colName]
            sub_collection = [x for x in wmb_collection.children if x.is_visible][0]
            arm_obj = sub_collection.collection.objects[0]
        else:
            arm_obj = targetCollection.collection
            sub_collection = targetCollection


        #arm_obj = sub_collection.collection # Maybe?

        material_remap = []
        mesh_name_to_material_index = {}

        if ("b2" not in arm_obj):
            arm_obj["b2"] = False # Assume not b2

        self.bayo_2 = arm_obj["b2"]
        if (self.bayo_2):
            self.ex_mat_A = arm_obj["exmatinfo"][0]
            self.ex_mat_B = arm_obj["exmatinfo"][1]

        i = 0
        '''sorted_objects = sorted(
            (obj for obj in getObjectChildren(arm_obj) if obj.type == 'MESH'),
            key=lambda obj: (int(obj.name.split("-")[0]), int(decimalFixup(obj.name.split("-")[2])))
        )'''

        for obj in getObjectChildren(arm_obj):
            if obj.type == 'MESH':
                for slot in obj.material_slots:
                    if slot.material:
                        if slot.material not in material_remap:
                            material_remap.append(slot.material)
                            i += 1
                        obj["material_id"] = material_remap.index(slot.material)

                        break

        def material_sort_key(mat):
            match = re.search(r'_(\d+)$', mat.name)
            if match:
                return (0, int(match.group(1)))
            return (1, 0)

        material_remap.sort(key=material_sort_key)

        print("Automatically remapping materials...")
        for i, material in enumerate(material_remap):
            print(f"[>] ID: {i:02} NAME: {material.name} / {material.bayo_data.shader}")



        '''# make up some shi
        if (self.bayo_2):
            for bone in arm_obj.data.bones:
                bone_name_to_id_map[bone.name] = bone["local_id"]

        else:
            current_highest_id = 0
            for bone in arm_obj.data.bones:
                if "id" in bone:
                    if bone["id"] > current_highest_id:
                        current_highest_id = bone["id"] 

            current_highest_id += 1 

            for bone in arm_obj.data.bones:
                if "id" not in bone:
                    bone["id"] = current_highest_id
                    current_highest_id+=1 

            for i, bone in enumerate(sorted(arm_obj.data.bones, key=lambda x: x["id"])):
                bone_name_to_id_map[bone.name] = i # Come up with some ids for local bones, these can be entirely arbitrary'''

        if (not EXPORT_AS_STATIC_MESH):
            if (self.bayo_2):
                bone_ready_list = []
                bones = arm_obj.data.bones
                highest_local_id = 0
                highest_global_id = 0

                for bone in bones:
                    if "local_id" in bone:
                        bone_name_to_id_map[bone.name] = bone["local_id"]
                        bone_ready_list.append(bone.name)
                        if bone["local_id"] > highest_local_id:
                            highest_local_id = bone["local_id"] + 1
                    if "id" in bone:
                        if bone["id"] > highest_global_id and bone["id"] < 1000:
                            highest_global_id = bone["id"] + 1

                print(f"Greatest Local ID: {highest_local_id}")
                print(f"Greatest Global ID: {highest_global_id}")

                existing_global_ids = set()
                seen_global_ids = set()
                duplicate_bones = []

                for bone in bones:
                    if "id" in bone:
                        if bone["id"] in seen_global_ids:
                            duplicate_bones.append(bone)
                        else:
                            seen_global_ids.add(bone["id"])
                            existing_global_ids.add(bone["id"])

                for bone in duplicate_bones:
                    while highest_global_id in existing_global_ids:
                        highest_global_id += 1
                    print(f"Duplicate global ID on '{bone.name}', reassigning to {highest_global_id}")
                    bone["id"] = highest_global_id
                    existing_global_ids.add(highest_global_id)
                    highest_global_id += 1

                for bone in bones:
                    if "id" not in bone:
                        while highest_global_id in existing_global_ids:
                            highest_global_id += 1
                        bone["id"] = highest_global_id
                        existing_global_ids.add(highest_global_id)
                        highest_global_id += 1

                if len(bone_ready_list) == 0:
                    for i, bone in enumerate(sorted(bones, key=lambda x: x["id"])):
                        bone_name_to_id_map[bone.name] = i
                else:
                    for bone in bones:
                        if bone.name not in bone_ready_list:
                            print(f"Assigning {bone.name} to Local ID {highest_local_id}")
                            bone_name_to_id_map[bone.name] = highest_local_id
                            highest_local_id += 1
            else:
                # make up some shi
                current_highest_id = 0
                for bone in arm_obj.data.bones:
                    if "id" in bone:
                        if bone["id"] > current_highest_id:
                            current_highest_id = bone["id"] + 1

                for bone in arm_obj.data.bones:
                    if "id" not in bone:
                        bone["id"] = current_highest_id
                        current_highest_id+=1 


                for i, bone in enumerate(sorted(arm_obj.data.bones, key=lambda x: x["id"])):
                    bone_name_to_id_map[bone.name] = i

        offset_ticker = 0
        self.header_offset = 0
        self.header_size = 128
        if (self.bayo_2):
            self.header_size = 192

        self.vtx_format = sub_collection.collection["vertex_format"]

        if (EXPORT_AS_STATIC_MESH):
            self.bone_count = 0
        else:
            self.bone_count = len(arm_obj.data.bones)

        self.has_that_one_fucked_vertex_format = False
        if self.vtx_format == 0x4000001F:
            self.has_that_one_fucked_vertex_format = True

        offset_ticker += self.header_size
        offset_ticker = align(offset_ticker, ALIGN_TARGET)

        bone_reference_dictionary = {}

        ## -- VERTEX CHUNK A --
        self.offset_vertexes = offset_ticker
        self.vertex_data = WMBVertexChunk(getObjectChildren(arm_obj), bone_reference_dictionary, self.bayo_2)
        offset_ticker += self.vertex_data.total_vertices * 32
        offset_ticker = align(offset_ticker, ALIGN_TARGET)

        self.offset_ex_vertexes = offset_ticker

        offset_ticker += self.vertex_data.total_vertices * self.vertex_data.exvertex_size
        offset_ticker = align(offset_ticker, ALIGN_TARGET)

        ## -- BONE CHUNK --
        if (not EXPORT_AS_STATIC_MESH):
            print("[>] Generating bone data...")

            self.offset_bone_parents = offset_ticker
            self.bone_parents = WMBBoneParents(arm_obj)

            offset_ticker += self.bone_count * 2
            offset_ticker = align(offset_ticker, ALIGN_TARGET)
            
            self.offset_bone_rel_positions = offset_ticker
            offset_ticker += self.bone_count * 12
            offset_ticker = align(offset_ticker, ALIGN_TARGET)
            self.offset_bone_abs_positions = offset_ticker

            self.bone_positions = WMBBonePosition(arm_obj)

            offset_ticker += self.bone_count * 12
            offset_ticker = align(offset_ticker, ALIGN_TARGET)

            self.offset_bone_index_translate_table = offset_ticker

            self.bone_index_translate_table = WMBBoneIndexTranslateTable(arm_obj)

            offset_ticker += self.bone_index_translate_table.size
            offset_ticker = align(offset_ticker, ALIGN_TARGET)

            self.bone_inverse_kinetic_table = WMBInverseKinetic(arm_obj)
            if (self.bone_inverse_kinetic_table.enabled):
                self.offset_bone_inverse_kinetic_table = offset_ticker
                offset_ticker += self.bone_inverse_kinetic_table.size
                offset_ticker = align(offset_ticker, ALIGN_TARGET)
            else:
                self.offset_bone_inverse_kinetic_table = 0

            self.bone_sym = WMBBoneSymmetries(arm_obj)
            if (self.bone_sym.enabled):
                self.offset_bone_sym = offset_ticker
                offset_ticker += self.bone_count * 2
                offset_ticker = align(offset_ticker, ALIGN_TARGET)
            else:
                self.offset_bone_sym = 0

            self.bone_flags = WMBBoneFlags(arm_obj)
            if (self.bone_flags.enabled):
                self.offset_bone_flags = offset_ticker
                offset_ticker += self.bone_count
                offset_ticker = align(offset_ticker, ALIGN_TARGET)
            else:
                self.offset_bone_flags = 0
        else:
            self.offset_bone_parents = 0
            self.offset_bone_rel_positions = 0
            self.offset_bone_abs_positions = 0
            self.offset_bone_index_translate_table = 0
            self.offset_bone_inverse_kinetic_table = 0
            self.offset_bone_sym = 0
            self.offset_bone_flags = 0

        if (self.bayo_2):
            self.exmat_blob = WMBExMaterialInfo(arm_obj, material_remap)
            self.exmat_offset = offset_ticker
            offset_ticker += (len(material_remap) * 16)
            offset_ticker = align(offset_ticker, ALIGN_TARGET)
            self.texture_list_offset = offset_ticker
            offset_ticker += (8 * len(self.exmat_blob.tex_info)) + 0x4

            offset_ticker = align(offset_ticker, ALIGN_TARGET)


        ## -- MATERIAL CHUNK --
        print("[>] Generating material data...")

        self.mat_blob = WMBMaterialBlob(arm_obj, material_remap, self.bayo_2)
        self.mat_offset_offset = offset_ticker

        offset_ticker += 4 * self.mat_blob.material_count
        offset_ticker = align(offset_ticker, ALIGN_TARGET)

        self.mat_offset = offset_ticker

        offset_ticker += self.mat_blob.total_size
        if (not self.bayo_2):
            offset_ticker = align(offset_ticker, ALIGN_TARGET)
        ## -- MESH CHUNK --
        print("[>] Generating mesh data...")
        self.mesh_blob = WMBMeshBlob(arm_obj)
        self.mesh_offset_offset = offset_ticker

        offset_ticker+=4*self.mesh_blob.mesh_count

        offset_ticker = align(offset_ticker, ALIGN_TARGET)
        self.mesh_offset = offset_ticker

        self.meshes = []
        self.mesh_offsets = []

        mesh_offset_ticker = 0

        sorted_children = sorted(
            (c for c in getObjectChildren(arm_obj) if c.type == 'MESH'),
            key=lambda obj: int(obj.name.split("-")[0])
        )

        for obj in sorted_children:
            name_parts = obj.name.split("-")
            if len(name_parts) == 3:
                if int(decimalFixup(name_parts[2])) == 0:
                    mesh_offset_ticker = align(mesh_offset_ticker, 32)
                    mesh_dat = WMBMesh(arm_obj, obj, bone_reference_dictionary, material_remap, self.bayo_2)
                    self.mesh_blob.offsets.append(mesh_offset_ticker)
                    self.mesh_offsets.append(mesh_offset_ticker)
                    self.meshes.append(mesh_dat)
                    mesh_offset_ticker+=mesh_dat.fetch_size()


def WMB0_Write_HDR(f, generated_data : WMBDataGenerator):
    f.write(b'WMB\x00')
    f.write(struct.pack("<i", 0))
    f.write(struct.pack("<i", generated_data.vtx_format))
    f.write(struct.pack("<i", generated_data.vertex_data.total_vertices))
    f.write(struct.pack("<b", generated_data.vertex_data.num_mapping))
    f.write(struct.pack("<b", generated_data.vertex_data.num_color))
    f.write(struct.pack("<h", 0))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", generated_data.offset_vertexes))
    f.write(struct.pack("<I", generated_data.offset_ex_vertexes))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 0))
    f.write(struct.pack("<I", 0))
    # Bone Chunk
    f.write(struct.pack("<I", generated_data.bone_count))
    f.write(struct.pack("<I", generated_data.offset_bone_parents))
    f.write(struct.pack("<I", generated_data.offset_bone_rel_positions))
    f.write(struct.pack("<I", generated_data.offset_bone_abs_positions))
    f.write(struct.pack("<I", generated_data.offset_bone_index_translate_table))
    # Material Chunk
    f.write(struct.pack("<I", generated_data.mat_blob.material_count))
    f.write(struct.pack("<I", generated_data.mat_offset_offset))
    f.write(struct.pack("<I", generated_data.mat_offset))
    # Mesh Chunk
    f.write(struct.pack("<I", generated_data.mesh_blob.mesh_count))
    f.write(struct.pack("<I", generated_data.mesh_offset_offset))
    f.write(struct.pack("<I", generated_data.mesh_offset))
    f.write(struct.pack("<I", 0))
    batch_count = 0
    for mesh in generated_data.meshes:
        batch_count += len(mesh.batches)

    f.write(struct.pack("<I", batch_count)) # Num Shader Setting

    f.seek(100)
    f.write(struct.pack("<I", generated_data.offset_bone_inverse_kinetic_table))
    f.write(struct.pack("<I", generated_data.offset_bone_sym))
    f.write(struct.pack("<I", generated_data.offset_bone_flags))

    if (generated_data.bayo_2):
        f.write(struct.pack("<I", generated_data.exmat_offset))
        f.write(struct.pack("<I", generated_data.texture_list_offset))
        f.write(struct.pack("<I", generated_data.ex_mat_A))
        f.write(struct.pack("<I", generated_data.ex_mat_B))

def WMB0_Write_VertexData(f, generated_data : WMBDataGenerator):
    for data in generated_data.vertex_data.vertex_infos:
        f.write(struct.pack("<fff", *data[0]))
        f.write(struct.pack("<ee", *data[5] ))

        if (generated_data.bayo_2):
            fx, fy, fz = data[1][0], data[1][1], data[1][2]
            f.write(struct.pack('<I', pack_b2_normal(fx, -fz, fy))) # Normals

        else:
            nx = int(round(data[1][0] * 127))
            ny = int(round(data[1][1] * 127))
            nz = int(round(data[1][2] * 127))

            nx = max(-127, min(127, nx))
            ny = max(-127, min(127, ny))
            nz = max(-127, min(127, nz))


            f.write(struct.pack('<4b', 0, ny, -nz, nx)) # Normals

        tx, ty, tz, d = data[2]
        tangent_bytes = bytes([
            tx,
            ty,
            tz,
            d
        ])

        f.write(tangent_bytes)

        if (EXPORT_AS_STATIC_MESH):
            f.write(struct.pack("<I", 0))
            #f.write(struct.pack("<BBBB", *data[7]))
            if (generated_data.vertex_data.num_mapping == 2):
                uv_bytes = float_to_half_bytes(data[1][0]) + float_to_half_bytes(1 - data[1][1])
                f.write(uv_bytes)
        else:
            f.write(struct.pack("<BBBB", *data[3])) # Bone Indexes
            f.write(struct.pack("<BBBB", *data[4])) # Bone Weights


    f.seek(generated_data.offset_ex_vertexes)
    for data in generated_data.vertex_data.exvertex_infos:
        f.write(struct.pack("<BBBB", *data[0]))
        if (generated_data.vertex_data.num_mapping == 2):
            f.write(struct.pack("<ee", *data[1] ))

def WMB0_Write_BoneParents(f, generated_data : WMBDataGenerator):
    bone_map = generated_data.bone_parents.bone_map

    max_index = max(bone_map.keys())

    for bone_index in range(max_index + 1):
        parent_index = bone_map.get(bone_index, -1)
        f.write(struct.pack("<h", parent_index))

def WMB0_Write_Positions_Rel(f, generated_data : WMBDataGenerator):
    bone_map = generated_data.bone_positions.bone_rel_map
    for bone_index in sorted(bone_map.keys()):
        position = bone_map[bone_index]
        f.write(struct.pack("<f", position[0]))
        f.write(struct.pack("<f", position[2]))
        f.write(struct.pack("<f", -position[1]))

def WMB0_Write_Positions_Abs(f, generated_data : WMBDataGenerator):
    bone_map = generated_data.bone_positions.bone_pos_map
    for bone_index in sorted(bone_map.keys()):
        position = bone_map[bone_index]
        f.write(struct.pack("<f", position[0]))
        f.write(struct.pack("<f", position[2]))
        f.write(struct.pack("<f", -position[1]))


def WMB0_Write_BoneIndexTranslateTable(f, generated_data : WMBDataGenerator):
    if (GENERATE_TRANSLATE_TABLE):
        for i in generated_data.bone_index_translate_table.data:
            f.write(struct.pack("<H", i))

    else:
        for i in generated_data.bone_index_translate_table.level_1:
            f.write(struct.pack("<h", i))
        for i in generated_data.bone_index_translate_table.level_2:
            f.write(struct.pack("<h", i))
        for i in generated_data.bone_index_translate_table.level_3:
            f.write(struct.pack("<h", i))

def WMB0_Write_IK(f, generated_data : WMBDataGenerator):
    if (generated_data.bone_inverse_kinetic_table.enabled):
        f.write(struct.pack("<b", generated_data.bone_inverse_kinetic_table.count))
        f.write(struct.pack("<bbb", *generated_data.bone_inverse_kinetic_table.data))
        f.write(struct.pack("<i", generated_data.bone_inverse_kinetic_table.offset))
        for table in generated_data.bone_inverse_kinetic_table.structures:
            f.write(struct.pack("<" + ("b" * 16), *table))

def WMB0_Write_Sym(f, generated_data : WMBDataGenerator):
    if (generated_data.bone_sym.enabled):
        bone_map = generated_data.bone_sym.sym_map

        max_index = max(bone_map.keys())
        for bone_index in range(max_index + 1):
            sym_idx = bone_map.get(bone_index, -1)
            f.write(struct.pack("<h", sym_idx))

def WMB0_Write_Flags(f, generated_data : WMBDataGenerator):
    if (generated_data.bone_flags.enabled):
        bone_map = generated_data.bone_flags.flag_map

        max_index = max(bone_map.keys())
        for bone_index in range(max_index + 1):
            sym_idx = bone_map.get(bone_index, -1)
            f.write(struct.pack("<B", sym_idx))

def WMB0_Write_Mat_Offsets(f, generated_data : WMBDataGenerator):
    for ofst in generated_data.mat_blob.offsets:
        f.write(struct.pack("<I", ofst))

def WMB0_Write_Mat(f, generated_data : WMBDataGenerator):
    for mat in generated_data.mat_blob.materials:
        mat.write(f)

def WMB0_Write_B2_ExMaterialInfo(f, generated_data : WMBDataGenerator):
    for mat in generated_data.mat_blob.materials:
        f.write(mat.shader_name.ljust(16, '\x00').encode('utf-8'))

    f.seek(generated_data.texture_list_offset)
    f.write(struct.pack("<I", len(generated_data.exmat_blob.tex_info)))
    for idx, flag in generated_data.exmat_blob.tex_info:
        f.write(struct.pack("<I", idx))
        f.write(struct.pack("<I", flag))


def WMB0_Write_Mesh_Offsets(f, generated_data : WMBDataGenerator):
    for ofst in generated_data.mesh_blob.offsets:
        f.write(struct.pack("<I", ofst))

def WMB0_Write_Mesh_Data(f, generated_data : WMBDataGenerator):
    batch_tick = 0
    for i, mesh in enumerate(generated_data.meshes):
        mesh_pos = generated_data.mesh_offset + generated_data.mesh_offsets[i]
        f.seek(mesh_pos)
        print(f"Writing mesh {mesh.name} @ {generated_data.mesh_offset + generated_data.mesh_offsets[i]}")

        f.write(struct.pack("<h", mesh.mesh_id))
        f.write(struct.pack("<h", mesh.batch_count))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<h", mesh.bounding_box_infos))
        f.write(struct.pack("<I", mesh.batch_offset_offset))
        f.write(struct.pack("<i", mesh.flags))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<i", 0))
        f.write(mesh.name.ljust(32, '\x00').encode('utf-8'))
        if (generated_data.bayo_2):
            f.write(struct.pack("<3f", *mesh.center))
            f.write(struct.pack("<f", mesh.height))
            f.write(struct.pack("<3f", *mesh.corner1))
            f.write(struct.pack("<3f", *mesh.corner2))

        else:
            f.write(struct.pack('<3ff3f3fff', *mesh.exdata))

        f.seek(mesh_pos + mesh.batch_offset_offset)
        print(f"|- Writing batch offsets {mesh_pos + mesh.batch_offset_offset}")
        for ofst in mesh.batch_offsets:
            f.write(struct.pack("<I", ofst))

        for i, batch in enumerate(mesh.batches):
            
            batch_offset = mesh_pos + mesh.batch_offset_offset + mesh.batch_offsets[i]
            print(f"   |- Writing batch @ {batch_offset}  | {len(batch.indices)} indice(s) @ {batch_offset + batch.indice_offset}")
            f.seek(batch_offset)
            f.write(struct.pack("<h", batch_tick))
            f.write(struct.pack("<h", batch.id))
            f.write(struct.pack("<H", batch.flags))
            f.write(struct.pack("<h", batch.exmaterial_id))
            f.write(struct.pack("<B", batch.material_id))
            f.write(struct.pack("<B", batch.has_bone_refs))
            f.write(struct.pack("<b", batch.unknownE1))
            f.write(struct.pack("<b", batch.unknownE2))
            f.write(struct.pack("<I", batch.vertex_start))
            f.write(struct.pack("<I", batch.vertex_end))
            f.write(struct.pack("<I", batch.primitive_type))
            f.write(struct.pack("<I", batch.indice_offset))
            f.write(struct.pack("<I", len(batch.indices)))
            f.write(struct.pack("<I", batch.vertex_offset))
            f.write(struct.pack("<IIIIIII", 0, 0, 0, 0, 0, 0, 0))
            f.write(struct.pack("<I", len(batch.required_bones)))

            if (USE_LARGE_BONES):
                f.write(struct.pack("<i", -1))
                for i in batch.required_bones:
                    f.write(struct.pack("<H", i))

            else:
                for i in batch.required_bones:
                    f.write(struct.pack("<B", i))

            f.seek(batch_offset + batch.indice_offset)
            for indice in batch.indices:
                f.write(struct.pack("<H", indice))

            batch_tick+=1


def export(filepath, op_inst=None, all_bone_refs=False, btt=True, large_bones=False, copy_uv=True, bayonetta_2=False, static_mesh=False, targetCol=None):
    global GENERATE_TRANSLATE_TABLE
    global USE_LARGE_BONES
    global OP_INSTANCE
    global EXCEPT_AFTER_GENERATION
    global COPY_UV_1_AS_2
    global EXPORT_AS_STATIC_MESH

    
    EXPORT_AS_STATIC_MESH = static_mesh
    USE_LARGE_BONES = large_bones
    GENERATE_TRANSLATE_TABLE = btt
    OP_INSTANCE = op_inst
    COPY_UV_1_AS_2 = copy_uv

    print("- BEGIN EXPORT -")
    f = open(filepath, "wb")
    print("[>] Preparing data...")

    generated_data = WMBDataGenerator(targetCollection=targetCol) # Loosely based off of MGR2Blender


    f.seek(generated_data.header_offset)
    WMB0_Write_HDR(f, generated_data)
    f.seek(generated_data.offset_vertexes)
    WMB0_Write_VertexData(f, generated_data)
    # -- RIGGED ONLY --
    if (not static_mesh):
        f.seek(generated_data.offset_bone_parents)
        WMB0_Write_BoneParents(f, generated_data)
        f.seek(generated_data.offset_bone_rel_positions)
        WMB0_Write_Positions_Rel(f, generated_data)
        f.seek(generated_data.offset_bone_abs_positions)
        WMB0_Write_Positions_Abs(f, generated_data)
        f.seek(generated_data.offset_bone_index_translate_table)
        WMB0_Write_BoneIndexTranslateTable(f, generated_data)
        f.seek(generated_data.offset_bone_inverse_kinetic_table)
        WMB0_Write_IK(f, generated_data)
        f.seek(generated_data.offset_bone_sym)
        WMB0_Write_Sym(f, generated_data)
        f.seek(generated_data.offset_bone_flags)
        WMB0_Write_Flags(f, generated_data) # I should have called this bone flags but oh well
    

    if (generated_data.bayo_2):
        f.seek(generated_data.exmat_offset)
        WMB0_Write_B2_ExMaterialInfo(f, generated_data)
    f.seek(generated_data.mat_offset_offset)
    WMB0_Write_Mat_Offsets(f, generated_data)
    f.seek(generated_data.mat_offset)
    WMB0_Write_Mat(f, generated_data)
    f.seek(generated_data.mesh_offset_offset)
    WMB0_Write_Mesh_Offsets(f, generated_data)
    f.seek(generated_data.mesh_offset)
    WMB0_Write_Mesh_Data(f, generated_data)
    f.seek(align(f.tell(), 32))
    f.write(b"This WMB was brought to you by Gaming With Portals, Raq, and Skyth")

    f.close()

    print("Done!")
    return {'FINISHED'}