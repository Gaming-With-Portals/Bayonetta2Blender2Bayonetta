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

    def read_s8_array(self, count):
        return struct.unpack((self.end_flag + "b"*count), self.f.read(count))

    def read_u8_array(self, count):
        return struct.unpack((self.end_flag + "B"*count), self.f.read(count))

    def read(self, len):
        return self.f.read(len)
    
    def read_f32_vector3(self):
        return struct.unpack(f"{self.end_flag}fff", self.f.read(12))