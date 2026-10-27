"""Analysis related to x86 floating point instructions.
x87 instructions start with D8 to DF; scalar SSE instructions use an F2 or F3 prefix.

We are interested in floating point constants, so we want to exclude instructions that:
- access the status register or environment (FLDCW, FLDENV)
- store a value (FST, FSTP)
- refer to integers (FI*)

Then filter on pointers into read-only sections.
"""

import re
import struct
from typing import Iterator, NamedTuple
from typing_extensions import Buffer
from reccmp.formats import PEImage

SCALAR_SSE_OPCODES = (0x10, 0x51, 0x58, 0x59, 0x5A, 0x5C, 0x5D, 0x5E, 0x5F)

SINGLE_PRECISION_OPCODES = frozenset(
    [
        (0xD8, 0x05),  # fadd
        (0xD8, 0x0D),  # fmul
        (0xD8, 0x15),  # fcom
        (0xD8, 0x1D),  # fcomp
        (0xD8, 0x25),  # fsub
        (0xD8, 0x2D),  # fsubr
        (0xD8, 0x35),  # fdiv
        (0xD8, 0x3D),  # fdivr
        (0xD9, 0x05),  # fld
    ]
    + [
        (0xF3, 0x0F, opcode, modrm)
        for opcode in SCALAR_SSE_OPCODES
        for modrm in range(0x05, 0x40, 0x08)
    ]
)

DOUBLE_PRECISION_OPCODES = frozenset(
    [
        (0xDC, 0x05),  # fadd
        (0xDC, 0x0D),  # fmul
        (0xDC, 0x15),  # fcom
        (0xDC, 0x1D),  # fcomp
        (0xDC, 0x25),  # fsub
        (0xDC, 0x2D),  # fsubr
        (0xDC, 0x35),  # fdiv
        (0xDC, 0x3D),  # fdivr
        (0xDD, 0x05),  # fld
    ]
    + [
        (0xF2, 0x0F, opcode, modrm)
        for opcode in SCALAR_SSE_OPCODES
        for modrm in range(0x05, 0x40, 0x08)
    ]
)

FLOAT_OPCODES = frozenset([*SINGLE_PRECISION_OPCODES, *DOUBLE_PRECISION_OPCODES])


# Match a superset of the floating point instructions above.
# Uses positive lookahead to support overlapping matches.
FLOAT_INSTRUCTION_RE = re.compile(
    rb"(?=((?:[\xd8\xd9\xdc\xdd]|[\xf2\xf3]\x0f[\x10\x51\x58\x59\x5a\x5c-\x5f])"
    rb"[\x05\x0d\x15\x1d\x25\x2d\x35\x3d].{4}))",
    flags=re.S,
)


class FloatInstruction(NamedTuple):
    # The address (or offset) of the instruction
    address: int
    # Instruction bytes before the absolute address operand
    opcode: tuple[int, ...]
    # The address used in the operand
    pointer: int


def find_float_instructions_in_buffer(
    buf: Buffer, base_addr: int = 0
) -> Iterator[FloatInstruction]:
    """Search the given binary blob for floating-point instructions that reference a pointer.
    If the base addr is given, add it to the offset of the instruction to get an absolute address.
    TODO: Uses `bytes` as the generic type for the Buffer protocol. See PEP 688 added in Python 3.12.
    """
    for match in FLOAT_INSTRUCTION_RE.finditer(buf):
        inst = match.group(1)
        opcode = tuple(inst[:-4])

        if opcode in FLOAT_OPCODES:
            (pointer,) = struct.unpack("<I", inst[-4:])
            yield FloatInstruction(base_addr + match.start(), opcode, pointer)


class FloatConstant(NamedTuple):
    address: int
    size: int
    value: float


def find_float_consts(image: PEImage) -> Iterator[FloatConstant]:
    """Floating point instructions that refer to a memory address can
    point to constant values. Search the code sections to find FP
    instructions and check whether the pointer address refers to
    read-only data."""

    # Multiple instructions can refer to the same float.
    # Return each float only once from this function.
    seen = set()

    const_regions = list(image.get_const_regions())

    for region in image.get_code_regions():
        for inst in find_float_instructions_in_buffer(region.data, region.addr):
            if inst.pointer in seen:
                continue

            seen.add(inst.pointer)

            # Make sure that the address of the operand is a relocation.
            if inst.address + len(inst.opcode) not in image.relocations:
                continue

            # Ignore instructions that point to variables
            size = 4 if inst.opcode in SINGLE_PRECISION_OPCODES else 8
            if any(
                inst.pointer in region.range and inst.pointer + size - 1 in region.range
                for region in const_regions
            ):
                (float_value,) = struct.unpack(
                    "<f" if size == 4 else "<d", image.read(inst.pointer, size)
                )
                yield FloatConstant(inst.pointer, size, float_value)
