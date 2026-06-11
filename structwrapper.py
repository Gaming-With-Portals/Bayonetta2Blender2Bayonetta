from io import BufferedReader
import struct

class BinReader:
    def __init__(self, f : BufferedReader, isBig=False):
        self.f = f
        self.big = isBig
        self.end_flag = "<"
        self.update_endianess_flag()

    def update_endianess_flag(self):
        if (self.big):
            self.end_flag = ">"
        else:
            self.end_flag = "<"

    def read_u32(self):
        return struct.unpack(f"{self.end_flag}I", self.f.read(4))[0]
    
    def read_u16(self):
        return struct.unpack(f"{self.end_flag}H", self.f.read(2))[0]
    
    def read_u8(self):
        return struct.unpack(f"{self.end_flag}B", self.f.read(1))[0]
    
    def read_s32(self):
        return struct.unpack(f"{self.end_flag}i", self.f.read(4))[0]
    
    def read_s16(self):
        return struct.unpack(f"{self.end_flag}h", self.f.read(2))[0]
    
    def read_s8(self):
        return struct.unpack(f"{self.end_flag}b", self.f.read(1))[0]

    def read_float32(self):
        return struct.unpack(f"{self.end_flag}f", self.f.read(4))[0]
    
    def seek(self, offset, whence=0):
        if (whence==0):
            self.f.seek(offset)
        elif (whence==1):
            self.f.seek(self.f.tell() + offset)

    def tell(self):
        return self.f.tell()

    def advance(self, bytes):
        self.f.read(bytes)

    def read_u32_array(self, count):
        return struct.unpack((self.end_flag + "I"*count), self.f.read(4 * count))

    def read_s16_array(self, count):
        return struct.unpack((self.end_flag + "h"*count), self.f.read(2 * count))

    def read_s8_array(self, count):
        return struct.unpack((self.end_flag + "b"*count), self.f.read(count))

    def read_u8_array(self, count):
        return struct.unpack((self.end_flag + "B"*count), self.f.read(count))

    def read(self, len):
        return self.f.read(len)
    
    def read_f32_vector3(self):
        return struct.unpack(f"{self.end_flag}fff", self.f.read(12))
    
    def read_rest(self):
        return self.f.read()

class BinWriter:
    def __init__(self, f : BufferedReader, isBig=False):
        self.f = f
        self.big = isBig
        self.end_flag = "<"
        self.update_endianess_flag()

    def update_endianess_flag(self):
        if (self.big):
            self.end_flag = ">"
        else:
            self.end_flag = "<"

    def write_u32(self, val):
        self.f.write(struct.pack(self.end_flag+"I", val))

    def write_s32(self, val):
        self.f.write(struct.pack(self.end_flag+"i", val))

    def write_u16(self, val):
        self.f.write(struct.pack(self.end_flag+"H", val))

    def write_s16(self, val):
        self.f.write(struct.pack(self.end_flag+"h", val))
        
    def write_u8(self, val):
        self.f.write(struct.pack(self.end_flag+"B", val))

    def write_s8(self, val):
        self.f.write(struct.pack(self.end_flag+"b", val))

    def write_float32(self, val):
        self.f.write(struct.pack(self.end_flag+"f", val))

    def write_float16(self, val):
        self.f.write(struct.pack(self.end_flag+"e", val))

    def write_vector3(self, x, y, z):
        self.f.write(struct.pack(self.end_flag+"fff", x, y, z))

    def write_packed_bytes(self, x, y, z, w):
        self.f.write(struct.pack(self.end_flag+"bbbb", x, y, z, w))

    def write_packed_bytes_unsigned(self, x, y, z, w):
        self.f.write(struct.pack(self.end_flag+"BBBB", x, y, z, w))

    def write(self, data):
        self.f.write(data)

    def tell(self):
        return self.f.tell()
    
    def seek(self, offset, whence=0):
        if (whence==0):
            self.f.seek(offset)
        elif (whence==1):
            self.f.seek(self.f.tell() + offset)