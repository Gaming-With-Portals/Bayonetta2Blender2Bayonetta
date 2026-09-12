#encoding = utf-8
import os
import sys

from ...utils.util import saveDatInfo
from ...utils.ioUtils import read_int32, read_uint32, read_uint16
from ...structwrapper import BinReader

def create_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def read_header(br : BinReader):
    Magic = br.read(4)
    if list(Magic) == [68, 65, 84, 0]:
        FileCount = br.read_u32()
        FileTableOffset = br.read_u32()
        ExtensionTableOffset = br.read_u32()
        NameTableOffset = br.read_u32()
        SizeTableOffset = br.read_u32()
        hashMapOffset = br.read_u32()
#         print(
# '''FileCount: %08x
# FileTableOffset: %08x
# ExtensionTableOffset:%08x
# NameTableOffset:%08x
# SizeTableOffset:%08x
# hashMapOffset:%08x
# '''%
#             (FileCount, FileTableOffset, ExtensionTableOffset,NameTableOffset,SizeTableOffset,hashMapOffset)
#         )
        return (FileCount, FileTableOffset, ExtensionTableOffset,NameTableOffset,SizeTableOffset,hashMapOffset)
    else:
        print('[-] wrong magic number detected')
        return False

def get_fileinfo(br : BinReader, index, FileTableOffset, ExtensionTableOffset, NameTableOffset, SizeTableOffset):
    br.seek(FileTableOffset + index * 4)
    FileOffset = br.read_u32()
    br.seek(ExtensionTableOffset + index * 4)
    Extension = br.read(4).decode('utf-8')
    br.seek(SizeTableOffset + index * 4)
    Size = br.read_u32()
    br.seek(NameTableOffset)
    FilenameAlignment = br.read_u32()
    i = 0
    while i < index:
        if list(br.read(FilenameAlignment))[FilenameAlignment-1] == 0:
            i += 1
    Filename = br.read(256).split(b'\x00')[0].decode('ascii')
#     print(
# '''
# FileIndex: %d
# Filename: %s
# FileOffset: %08x
# Size: %08x
# Extension: %s'''%
# (index,Filename,FileOffset,Size,Extension)
#         )
    return index,Filename,FileOffset,Size,Extension

def extract_file(fp, filename, FileOffset, Size, extract_dir):
    create_dir(extract_dir)
    fp.seek(FileOffset)
    FileContent = fp.read(Size)
    with open(extract_dir +filename,'wb') as outfile:
        # print("extracting file %s to %s/%s"%(filename,extract_dir,filename))
        outfile.write(FileContent)
    if filename.find('wtp') > -1 and False:  # Removed due to not needed anymore when using Blender DTT import.
        wtp_fp = open(extract_dir + '/'+filename,"rb")
        content = wtp_fp.read(Size)
        dds_group = content.split(b'DDS ')
        dds_group = dds_group[1:]
        for i in range(len(dds_group)):
            print("unpacking %s to %s/%s"%(filename,extract_dir ,filename.replace('.wtp','_%d.dds'%i)))
            dds_fp = open(extract_dir + '/'+filename.replace('.wtp','_%d.dds'%i), "wb")
            dds_fp.write(b'DDS ')
            dds_fp.write(dds_group[i])
            dds_fp.close()
        wtp_fp.close()
        #os.remove("%s/%s"%(extract_dir,filename))
    # print("done")

def get_all_files(path):
    pass

def extract_hashes(fp, extract_dir, FileCount, hashMapOffset, fileNamesOffset, filename):
    create_dir(extract_dir)

    # file_order.metadata
    # Filename Size
    fp.seek(fileNamesOffset)
    fileNameSize = read_int32(fp)

    # Filenames
    fileNames = []
    for i in range(FileCount):
        fileNames.append(fp.read(fileNameSize))

    # Write filenames to json
    saveDatInfo(
        extract_dir + '/dat_info.json',
        [file.decode('utf-8').rstrip('\x00') for file in fileNames],
        filename
    )

    # hash_data.metadata
    # Header
    fp.seek(hashMapOffset)
    preHashShift = read_uint32(fp)
    bucketOffsetsOffset = read_uint32(fp)
    hashesOffset = read_uint32(fp)
    fileIndicesOffset = read_uint32(fp)

    # Bucket Offsets
    fp.seek(hashMapOffset + bucketOffsetsOffset)
    bucketOffsets = []
    while fp.tell() < (hashMapOffset + hashesOffset):
        bucketOffsets.append(read_uint16(fp))

    # Hashes
    fp.seek(hashMapOffset + hashesOffset)
    hashes = []
    for i in range(FileCount):
        hashes.append(fp.read(4))

    # File Indices
    fp.seek(hashMapOffset + fileIndicesOffset)
    fileIndices = []
    for i in range(FileCount):
        fileIndices.append(read_uint16(fp))

def main(filename, extract_dir, ROOT_DIR):

    with open(filename,"rb") as fp:
        br = BinReader(fp)
        br.seek(0x4)
        if (br.read_u32() > 50000):
            br.big = True
            br.update_endianess_flag()
            print("[!] Big Endian!")

        br.seek(0)


        headers = read_header(br)

        if headers:
            FileCount, FileTableOffset, ExtensionTableOffset,NameTableOffset,SizeTableOffset,hashMapOffset = headers

            extractedFiles = 0
            for i in range(FileCount):
                extract_dir_sub = ''
                index,Filename,FileOffset,Size,Extension = get_fileinfo(br, i, FileTableOffset,ExtensionTableOffset, NameTableOffset,SizeTableOffset)
                if extract_dir != '':
                    extract_dir_sub = os.path.join(extract_dir, filename.replace(ROOT_DIR ,''))
                    extract_file(fp, Filename, FileOffset, Size, extract_dir_sub)
                    extractedFiles += 1
            
            extract_hashes(fp, extract_dir, FileCount, hashMapOffset, NameTableOffset, os.path.basename(filename))
            print(f"[+] {extractedFiles} files extracted from {filename}")
        else:
            FileCount = 0
    if (FileCount):
        return Filename
    return False


if __name__ == '__main__':
    extract_dir = ''
    dirname = ''
    usage = "\nUsage:\npython dat_unpacker.py your_dat_path your_extract_path"
    usage1 = "\nUsage:\nblender --background --python dat_unpacker.py your_dat_path your_extract_path"
    if len(sys.argv) < 3:
        print(usage)
        exit()
    if len(sys.argv) > 2:
        dir_name = sys.argv[1]
        extract_dir = sys.argv[2]
        print()
        if os.path.split(sys.argv[0])[-1].lower().find("blender") >-1:
            if len(sys.argv) < 6:
                print(usage1)
                exit()
            dir_name = sys.argv[4]
            extract_dir = sys.argv[5]
        if not os.path.exists(extract_dir):
            create_dir(extract_dir)
    ROOT_DIR = dir_name
    for dirpath,dirnames,filename in os.walk(dir_name):
        for file in filename:
            filename = os.path.join(dirpath,file)
            main(filename, extract_dir, ROOT_DIR)
