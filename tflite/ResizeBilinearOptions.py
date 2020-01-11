# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class ResizeBilinearOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsResizeBilinearOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ResizeBilinearOptions()
        x.Init(buf, n + offset)
        return x

    # ResizeBilinearOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ResizeBilinearOptions
    def NewHeight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ResizeBilinearOptions
    def NewWidth(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def ResizeBilinearOptionsStart(builder): builder.StartObject(2)
def ResizeBilinearOptionsAddNewHeight(builder, newHeight): builder.PrependInt32Slot(0, newHeight, 0)
def ResizeBilinearOptionsAddNewWidth(builder, newWidth): builder.PrependInt32Slot(1, newWidth, 0)
def ResizeBilinearOptionsEnd(builder): return builder.EndObject()
