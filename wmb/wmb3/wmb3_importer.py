
from ...structwrapper import BinReader
import os
import numpy as np
import bpy
import math

DEVASTATION_MDL = False
VERTEX_FORMAT = 0

def read_cstring(f):
    buf = bytearray()
    while True:
        b = f.read(1)
        if not b or b == b"\x00":
            break
        buf += b
    return buf.decode("utf-8", errors="replace")

class WMB3VertexGroup:
    def __init__(self, f : BinReader):
        self.vertexOffset = f.read_u32()
        self.vertexExDataOffset = f.read_u32()
        f.advance(8)
        self.vertexSize = f.read_u32()
        self.vertexExSize = f.read_u32()
        f.advance(8)
        self.numVertexes = f.read_u32()
        if (not DEVASTATION_MDL):
            self.vertexFlags = f.read_u32()
        
        self.indexBufferOffset = f.read_u32()
        self.numIndexes = f.read_u32()
        self.linkedGroup = None

        self.vertexData = []
        self.vertexExData = []
        self.indexes = []

    def LinkWME3(self, group):
        self.linkedGroup = group

    def ReadVertexInfo(self, wmb_flags, f : BinReader, wme_f : BinReader):
        vtxOffset = self.vertexOffset
        exVtxOffset = self.vertexExDataOffset
        indexOffset = self.indexBufferOffset

        vertexReader = f
        indexReader = f
        exVertexReader = f

        if (wme_f != None):
            if (vtxOffset == 0):
                vtxOffset = self.linkedGroup.vertexOffset
                vertexReader = wme_f
            if (exVtxOffset == 0):
                exVtxOffset = self.linkedGroup.vertexExDataOffset
                exVertexReader = wme_f
            if (indexOffset == 0):
                indexOffset = self.linkedGroup.indexBufferOffset
                indexReader = wme_f

        if (vtxOffset == 0):
            print("[!] Failed to read Vertex Buffer")
        if (exVtxOffset == 0):
            print("[!] Failed to read ExVertex Buffer")
        if (indexOffset == 0):
            print("[!] Failed to read Index Buffer")

        vertexAttributes = []
        exVertexAttributes = []

        if (not DEVASTATION_MDL):
            if self.vertexFlags == 0xe:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('uv_2', 'f2', 2))
                vertexAttributes.append(('col_1', 'u1', 4))

                # EX
                exVertexAttributes.append(('normals', 'f2', 4))
                exVertexAttributes.append(('uv_3', 'f2', 2))
                exVertexAttributes.append(('uv_4', 'f2', 2))
            elif self.vertexFlags == 0xc:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('uv_2', 'f2', 2))
                vertexAttributes.append(('col_1', 'u1', 4))

                # EX
                exVertexAttributes.append(('normals', 'f2', 4))
                exVertexAttributes.append(('uv_3', 'f2', 2))
                exVertexAttributes.append(('uv_4', 'f2', 2))
                exVertexAttributes.append(('uv_5', 'f2', 2))
            elif self.vertexFlags == 0xb:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('bone_index', 'u1', 4))
                vertexAttributes.append(('bone_weight', 'u1', 4))

                # EX
                exVertexAttributes.append(('uv_2', 'f2', 2))
                exVertexAttributes.append(('col_1', 'u1', 4))
                exVertexAttributes.append(('normals', 'f2', 4))
                exVertexAttributes.append(('uv_3', 'f2', 2))
            elif self.vertexFlags == 0xa:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('bone_index', 'u1', 4))
                vertexAttributes.append(('bone_weight', 'u1', 4))

                # EX
                exVertexAttributes.append(('uv_2', 'f2', 2))
                exVertexAttributes.append(('col_1', 'u1', 4))
                exVertexAttributes.append(('normals', 'f2', 4))
            elif self.vertexFlags == 0x8:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('bone_index', 'u1', 4))
                vertexAttributes.append(('bone_weight', 'u1', 4))

                # EX
                exVertexAttributes.append(('uv_2', 'f2', 2))
                exVertexAttributes.append(('normals', 'f2', 4))
                exVertexAttributes.append(('uv_3', 'f2', 2))
            elif self.vertexFlags == 0x7:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('bone_index', 'u1', 4))
                vertexAttributes.append(('bone_weight', 'u1', 4))

                # EX
                exVertexAttributes.append(('uv_2', 'f2', 2))
                exVertexAttributes.append(('normals', 'f2', 4))
            elif self.vertexFlags == 0x5:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('uv_2', 'f2', 2))
                vertexAttributes.append(('col_1', 'u1', 4))

                # EX
                exVertexAttributes.append(('normals', 'f2', 4))
                exVertexAttributes.append(('uv_3', 'f2', 2))
            elif self.vertexFlags == 0x4:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('uv_2', 'f2', 2))
                vertexAttributes.append(('col_1', 'u1', 4))

                # EX
                exVertexAttributes.append(('normals', 'f2', 4))
            elif self.vertexFlags == 0x3:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('col_1', 'u1', 4))

                # EX
                exVertexAttributes.append(('normals', 'f2', 4))
            elif self.vertexFlags == 0x1:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('uv_2', 'f2', 2))

                # EX
                exVertexAttributes.append(('normals', 'f2', 4))
            elif self.vertexFlags == 0x0:
                # CORE
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('tangents', 'u1', 4))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('normals', 'f2', 4))

                exVtxOffset = 0
            else:
                print(f"[!] Unknown vertex flag: 0x{self.vertexFlags:x} - {self.vertexSize}")
        else:
            print("[>] Using VertexFormat...")
            if (VERTEX_FORMAT == 65847):
                vertexAttributes.append(('position', 'f4', 3))
                vertexAttributes.append(('uv_1', 'f2', 2))
                vertexAttributes.append(('tangents', 'u1', 4))

                exVertexAttributes.append(('normals', 'f2', 4))
                exVertexAttributes.append(('bone_index', 'u1', 4))
                exVertexAttributes.append(('bone_weight', 'u1', 4))
                exVertexAttributes.append(('col_1', 'u1', 4))

                



        if (vtxOffset != 0):
            vertexReader.seek(vtxOffset)
            dt = np.dtype(vertexAttributes)
            self.vertexData = np.fromfile(vertexReader.f, dt, self.numVertexes)
        if (exVtxOffset != 0):
            exVertexReader.seek(exVtxOffset)
            dt = np.dtype(exVertexAttributes)
            self.vertexExData = np.fromfile(exVertexReader.f, dt, self.numVertexes)
        if (indexOffset != 0):
            indexReader.seek(indexOffset)
            if (not DEVASTATION_MDL):
                if (wmb_flags & 0x8):
                    self.indexes = indexReader.read_u32_array(self.numIndexes)
                else:
                    self.indexes = indexReader.read_u16_array(self.numIndexes)
            else:
                self.indexes = indexReader.read_u16_array(self.numIndexes)

        
        


class WMB3Batch:
    def __init__(self, f : BinReader):
        self.vertexGroupIndex = f.read_u32()
        if (not DEVASTATION_MDL):
            self.boneSetIndex = f.read_u32()
        else:
            self.boneSetIndex = 0

        self.vertexStart = f.read_u32()
        self.indexStart = f.read_u32()
        self.numVertexes = f.read_u32()
        self.numIndexes = f.read_u32()
        if (not DEVASTATION_MDL):
            self.numPrims = f.read_u32() # indexes divided by 3
        else:
            self.numPrims = self.numIndexes*3 # Generate

class WMB3Mesh:
    def __init__(self, f : BinReader):
        self.nameOffset = f.read_u32()
        self.boundingBox = (f.read_f32_vector3(), f.read_f32_vector3())
        if (DEVASTATION_MDL):
            self.groups = []
            for _ in range(5):
                self.groups.append((f.read_u32(), f.read_u32()))


        self.offsetMaterials = f.read_u32()
        self.numMaterials = f.read_u32()

        if (not DEVASTATION_MDL):
            self.offsetBones = f.read_u32()
            self.numBones = f.read_u32()

        pos = f.tell()
        f.seek(self.nameOffset)
        self.name = read_cstring(f.f)
        f.seek(self.offsetMaterials)
        self.materials = f.read_s16_array(self.numMaterials)
        if (not DEVASTATION_MDL):
            f.seek(self.offsetBones)
            self.bones = f.read_s16_array(self.numBones)
        else:
            self.bones = []

        f.seek(pos)
        print(f"[>] Mesh \"{self.name}\"")

class WMB3BatchInfo:
    def __init__(self, f : BinReader):
        self.vertexGroupID = f.read_s32()
        self.meshIndex = f.read_s32()
        self.materialIndex = f.read_s32()
        self.colTreeNodeIndex = f.read_s32()
        self.meshMatPairIndex = f.read_s32()
        self.indexToUnknown1 = f.read_s32()



class WMB3LOD:
    def __init__(self, f : BinReader):
        self.offsetName = f.read_u32()
        self.lodLevel = f.read_u32()
        self.batchStart = f.read_u32()
        self.offsetBatchInfo = f.read_u32()
        self.numBatchInfo = f.read_u32()
        self.batchInfos = []
        pos = f.tell()
        f.seek(self.offsetName)
        self.name = read_cstring(f.f)

        f.seek(self.offsetBatchInfo)
        for _ in range(self.numBatchInfo):
            self.batchInfos.append(WMB3BatchInfo(f))

        f.seek(pos)
        


class WME3File:
    def __init__(self, filepath):
        print("[>] Importing WME3... ", end="")
        rf = open(filepath, "rb")
        f = BinReader(rf)
        self.tag = f.read(4).decode()

        if (self.tag != "WME3"):
            print(f"\n[!] Couldn't load WME3 (not a WME3 file but instead {self.tag})")
            return

        f.advance(12)

        self.vertexGroupOffset = f.read_u32()
        self.numVertexGroup = f.read_u32()
        self.lodMapOffset = f.read_u32()
        self.lodCount = f.read_u32()

        self.vertexGroups = []
        f.seek(self.vertexGroupOffset)
        for i in range(self.numVertexGroup):
            self.vertexGroups.append(WMB3VertexGroup(f))

        print(f"[OK, LOADED {len(self.vertexGroups)} VERTEX GROUP(S)]")

        rf.close()
        self.fp = filepath





def ImportWMB3(filepath, target_col="WMB"):
    global DEVASTATION_MDL
    global VERTEX_FORMAT

    folder = os.path.dirname(filepath)
    wme_path = os.path.join(folder, os.path.splitext(filepath)[0] + ".wme")
    wmeFile = None

    if (os.path.exists(wme_path)):
        wmeFile = WME3File(wme_path)



    rf = open(filepath, "rb")
    f = BinReader(rf, False)

    if (f.read(4) != b'WMB3'):
        print("[!] Not a WMB3 file!") # impossible probably
        return
    
    if (target_col=="WMB"):
        if ("WMB" not in bpy.data.collections):
            wmb_collection = bpy.data.collections.new("WMB")
            bpy.context.scene.collection.children.link(wmb_collection)
        else:
            wmb_collection = bpy.data.collections["WMB"]
    else:
        wmb_collection = bpy.data.collections[target_col]
    
    print("[>] Importing WMB3...")
    
    version = f.read_u32()

    devastation = False

    if (version == 65536):
        devastation = True


    if (not devastation):
        unknownA = f.read_u32()
        flags = f.read_s16()
        referenceBone = f.read_s16()
        bb_xyz = f.read_f32_vector3()
        bb_uvw = f.read_f32_vector3()

        offsetBones = f.read_u32()
        numBones = f.read_u32()
        offsetBoneIndexTranslateTable = f.read_u32()
        boneTranslateTableSize = f.read_u32()
        offsetVertexGroups = f.read_u32()
        numVertexGroups = f.read_u32()
        offsetBatches = f.read_u32()
        numBatches = f.read_u32()
        offsetLods = f.read_u32()
        numLods = f.read_u32()
        offsetColTreeNodes = f.read_u32()
        numColTreeNodes = f.read_u32()
        offsetBoneMap = f.read_u32()
        boneMapSize = f.read_u32()
        offsetBoneSets = f.read_u32()
        numBoneSets = f.read_u32()
        offsetMaterials = f.read_u32()
        numMaterials = f.read_u32()
        offsetMeshes = f.read_u32()
        numMeshes = f.read_u32()
        offsetMeshMaterial = f.read_u32()
        numMeshMaterial = f.read_u32()
        offsetUnknown0 = f.read_u32()
        numUnknown0 = f.read_u32()
    else:
        unknownA = f.read_u32()
        vertexFormat = f.read_u32()
        unknownCount = f.read_s16()
        unknownTerminator = f.read_s16()
        bb_xyz = f.read_f32_vector3()
        bb_uvw = f.read_f32_vector3()

        offsetVertexGroups = f.read_u32()
        numVertexGroups = f.read_u32()
        offsetBatches = f.read_u32()
        numBatches = f.read_u32()
        offsetBatchGroups = f.read_u32()
        offsetBones = f.read_u32()
        numBones = f.read_u32()
        offsetBoneIndexTranslateTable = f.read_u32()
        boneTranslateTableSize = f.read_u32()
        offsetBoneSets = f.read_u32()
        numBoneSets = f.read_u32()
        offsetMaterials = f.read_u32()
        numMaterials = f.read_u32()
        offsetTextureIDs = f.read_u32()
        numTextureIDs = f.read_u32()
        offsetMeshes = f.read_u32()
        numMeshes = f.read_u32()

        VERTEX_FORMAT = vertexFormat



    wmb_name = os.path.splitext(os.path.basename(filepath))[0]

    model_collection = bpy.data.collections.new(wmb_name)

    model_collection["wmb_ver"] = 3
    if not devastation:
        model_collection["flags"] = flags
    else:
        model_collection["flags"] = 0
        flags = -1
        model_collection["special_type"] = "transformers_devastation"
        
    wmb_collection.children.link(model_collection)

    DEVASTATION_MDL = devastation
    

    if numBones > 0:
        arm_data = bpy.data.armatures.new(wmb_name)
        arm_obj = bpy.data.objects.new(wmb_name, arm_data)
        model_collection.objects.link(arm_obj)


    print("[>] Reading vertex groups...")
    vertexGroups = []
    f.seek(offsetVertexGroups)
    xr = None
    if (wmeFile != None):
        x = open(wme_path, "rb")
        xr = BinReader(x)

    for i in range(numVertexGroups):
        vtxGroup = WMB3VertexGroup(f)
        if (wmeFile != None):
            if (len(wmeFile.vertexGroups) != numVertexGroups):
                print("[!] Invalid WME3 File! (Mismatched vertex groups)")
                print("Disabling WME linking, you may get an error later.")
                wmeFile = None

            vtxGroup.LinkWME3(wmeFile.vertexGroups[i])

        pos = f.tell()
        vtxGroup.ReadVertexInfo(flags, f, xr)
        f.seek(pos)
        vertexGroups.append(vtxGroup)

    print("[>] Reading batches...")
    batches = []
    f.seek(offsetBatches)
    for i in range(numBatches):
        batches.append(WMB3Batch(f))

    print("[>] Reading meshes...")
    meshes = []
    f.seek(offsetMeshes)
    for i in range(numMeshes):
        meshes.append(WMB3Mesh(f))


    if (not DEVASTATION_MDL):
        print("[>] Reading LODs...")
        f.seek(offsetLods)
        lods = []
        for i in range(numLods):
            lods.append(WMB3LOD(f))
    else:
        print("[!] Skipping LOD Read!")


    print("[>] Generating geometry...")
    '''for i, batch in enumerate(batches):
        print(f"[>] Batch {i+1}/{len(batches)}")
        vtxGroup = vertexGroups[batch.vertexGroupIndex]'''


    if (not DEVASTATION_MDL):
        for lod in lods:
            lodBatches = batches[lod.batchStart:lod.batchStart+lod.numBatchInfo]
            print(f"[>] Loading {lod.name} from {len(lodBatches)} related batches")
            
            for i, batch in enumerate(lodBatches):
                batchInfo = lod.batchInfos[i]
                vtxGroup = vertexGroups[batch.vertexGroupIndex] # It doesn't matter if you use batch or batchindex for this, they are the same
                mesh = meshes[batchInfo.meshIndex]

                batchVertexes = vtxGroup.vertexData[batch.vertexStart:batch.vertexStart+batch.numVertexes]
                batchExVertexes = vtxGroup.vertexExData[batch.vertexStart:batch.vertexStart+batch.numVertexes]
                batchIndexes = vtxGroup.indexes[batch.indexStart:batch.indexStart+batch.numIndexes]

                object_name = f"{batchInfo.meshIndex}-{mesh.name}-{i}-{lod.lodLevel}"

                mesh = bpy.data.meshes.new(object_name)
                obj = bpy.data.objects.new(object_name, mesh)
                obj.rotation_euler = (math.radians(90), 0, 0)
                model_collection.objects.link(obj)

                mesh.vertices.add(batch.numVertexes)
                mesh.vertices.foreach_set('co', batchVertexes['position'].astype(np.float32).ravel())


                numTris = batch.numIndexes // 3
                mesh.loops.add(numTris * 3)
                mesh.polygons.add(numTris)

                mesh.polygons.foreach_set('loop_start', np.arange(0, numTris * 3, 3))
                mesh.polygons.foreach_set('loop_total', np.full(numTris, 3))
                mesh.loops.foreach_set('vertex_index', batchIndexes)


                mesh.update()

                if (lod.lodLevel != 0):
                    obj.hide_set(True)
    else:
        print("[>] Reading Devastation Data...")
        # Devastation Importer
        f.seek(offsetBatchGroups)
        batchGroups = []
        for _ in range(5):
            batchGroups.append((f.read_u32(), f.read_u32())) # Group Offset, Count


        batchGroupBatches = []
        for group in batchGroups:
            batchDatas = []
            f.seek(group[0])
            for _ in range(group[1]):
                batchDatas.append((f.read_u32(), f.read_u32(), f.read_u16(), f.read_u16(), f.read_u32()))

            batchGroupBatches.append(batchDatas)

            print(f"[>] Batch Group: {len(batchDatas)}")

        for meshID, dataMesh in enumerate(meshes):
            for groupID, group in enumerate(dataMesh.groups):
                print(f"[>] Reading mesh group {group[0]}, {group[1]}")
                if (group[1] == 0):
                    print("[!] Empty! Skipping...")
                    continue

                f.seek(group[0])
                batchGroupIndices = f.read_u16_array(group[1])

                for batchIndx in batchGroupIndices:
                    batchData = batchGroupBatches[groupID][batchIndx]
                    batch = batches[batchData[0]]

                    
                    
                    vtxGroup = vertexGroups[batch.vertexGroupIndex] # It doesn't matter if you use batch or batchindex for this, they are the same
                    batchVertexes = vtxGroup.vertexData[batch.vertexStart:batch.vertexStart+batch.numVertexes]
                    batchExVertexes = vtxGroup.vertexExData[batch.vertexStart:batch.vertexStart+batch.numVertexes]
                    batchIndexes = vtxGroup.indexes[batch.indexStart:batch.indexStart+batch.numIndexes]

                    object_name = f"{meshID}-{dataMesh.name}-{batchIndx}-{groupID}"

                    mesh = bpy.data.meshes.new(object_name)
                    obj = bpy.data.objects.new(object_name, mesh)
                    obj.rotation_euler = (math.radians(90), 0, 0)
                    model_collection.objects.link(obj)

                    mesh.vertices.add(batch.numVertexes)
                    mesh.vertices.foreach_set('co', batchVertexes['position'].astype(np.float32).ravel())


                    numTris = batch.numIndexes // 3
                    mesh.loops.add(numTris * 3)
                    mesh.polygons.add(numTris)

                    mesh.polygons.foreach_set('loop_start', np.arange(0, numTris * 3, 3))
                    mesh.polygons.foreach_set('loop_total', np.full(numTris, 3))
                    mesh.loops.foreach_set('vertex_index', batchIndexes)


                    mesh.update()


                
        




    return {"FINISHED"}